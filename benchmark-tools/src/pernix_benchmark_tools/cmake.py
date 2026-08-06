from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CMakeProject:
    """Configuration for a CMake project."""

    source_dir: Path
    build_dir: Path

    build_type: str = "Release"
    generator: str | None = "Ninja"

    install_prefix: Path | None = None
    jobs: int | None = None

    definitions: dict[str, str | bool | int | float | Path] = field(
        default_factory=dict[str, str | bool | int | float | Path]
    )

    cmake_arguments: list[str] = field(default_factory=list[str])
    build_arguments: list[str] = field(default_factory=list[str])

    cmake_executable: Path | None = None
    generator_executable: Path | None = None

    environment: dict[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        self.source_dir = Path(self.source_dir).resolve()
        self.build_dir = Path(self.build_dir).resolve()

        if self.install_prefix is not None:
            self.install_prefix = Path(self.install_prefix).resolve()

    @property
    def effective_jobs(self):
        return self.jobs or os.cpu_count() or 1

    def configure(
        self,
        *,
        fresh: bool = False,
        extra_arguments: Sequence[str] = (),
    ) -> None:
        """Configure the CMake project."""

        self._validate_source_dir()

        if fresh:
            self.clean()

        self.build_dir.mkdir(parents=True, exist_ok=True)

        command: list[str] = [
            str(self._find_cmake()),
            "-S",
            str(self.source_dir),
            "-B",
            str(self.build_dir),
            "-G",
            self.generator if self.generator is not None else "",
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            f"-DCMAKE_MAKE_PROGRAM={self._find_generator_executable()}",
        ]

        if self.install_prefix is not None:
            command.append(f"-DCMAKE_INSTALL_PREFIX={self.install_prefix}")

        command.extend(
            f"-D{name}={self._format_definition(value)}"
            for name, value in self.definitions.items()
        )

        command.extend(self.cmake_arguments)
        command.extend(extra_arguments)

        self._run(command)

    def build(
        self,
        *,
        target: str | None = None,
        configure: bool = True,
        extra_arguments: Sequence[str] = (),
    ) -> None:
        """Build the CMake project."""

        if configure and not self.is_configured():
            self.configure()

        command: list[str] = [
            str(self._find_cmake()),
            "--build",
            str(self.build_dir),
            "--parallel",
            str(self.effective_jobs),
        ]

        if target is not None:
            command.extend(["--target", target])

        command.extend(self.build_arguments)

        if extra_arguments:
            command.append("--")
            command.extend(extra_arguments)

        self._run(command)

    def configure_and_build(
        self,
        *,
        target: str | None = None,
        fresh: bool = False,
    ) -> None:
        """Configure and build the CMake project."""

        self.configure(fresh=fresh)
        self.build(target=target)

    def install(self, *, prefix: Path | None = None, configure: bool = True) -> None:
        """Install the CMake project."""

        if configure and not self.is_configured():
            self.configure()

        command = [
            str(self._find_cmake()),
            "--install",
            str(self.build_dir),
            "--config",
            self.build_type,
        ]

        effective_prefix = prefix or self.install_prefix

        if effective_prefix is not None:
            command.extend(["--prefix", str(Path(effective_prefix).resolve())])

        self._run(command)

    def test(
        self,
        *,
        configure: bool = True,
        output_on_failure: bool = True,
        extra_arguments: Sequence[str] = (),
    ) -> None:
        """Run tests for the CMake project."""

        if configure and not self.is_configured():
            self.configure()

        command: list[str] = [
            str(self._find_ctest()),
            "--test-dir",
            str(self.build_dir),
            "--build-config",
            self.build_type,
        ]

        if output_on_failure:
            command.append("--output-on-failure")

        command.extend(extra_arguments)
        self._run(command)

    def clean(self) -> None:
        """Clean the build directory."""
        if self.build_dir.exists():
            logger.info("Cleaning build directory: %s", self.build_dir)
            shutil.rmtree(self.build_dir)

    def clean_target(self) -> None:
        """Run the generated clean target in the build directory."""

        if not self.is_configured():
            return

        self.build(target="clean", configure=False)

    def is_configured(self) -> bool:
        """Check if the project is configured."""

        return (self.build_dir / "CMakeCache.txt").is_file()

    def _validate_source_dir(self) -> None:
        cmake_lists = self.source_dir / "CMakeLists.txt"

        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"Source directory does not exist: {self.source_dir}"
            )

        if not cmake_lists.is_file():
            raise FileNotFoundError(f"CMakeLists.txt not found: {cmake_lists}")

    def _find_cmake(self) -> Path:
        """Find the CMake executable."""

        if self.cmake_executable is not None:
            return self._validate_executable(self.cmake_executable, "CMake")

        executable = shutil.which("cmake")

        if executable is None:
            raise FileNotFoundError("CMake executable not found in PATH")

        return self._validate_executable(Path(executable), "CMake")

    def _find_generator_executable(self) -> Path | None:
        """Find the generator executable if specified."""

        if self.generator_executable is not None:
            return self._validate_executable(self.generator_executable, "Generator")

        if self.generator is None:
            return None

        executable_name = {
            "Ninja": "ninja",
            "Ninja Multi-Config": "ninja",
        }.get(self.generator, self.generator)

        executable = shutil.which(executable_name)

        if executable is None:
            raise FileNotFoundError(
                f"{self.generator} executable ({executable_name}) not found in PATH"
            )

        return self._validate_executable(Path(executable), "Generator")

    def _find_ctest(self) -> Path:
        """Find the CTest executable accompanying CMake."""

        executable = self._find_cmake().with_name("ctest")

        if executable.is_file():
            return self._validate_executable(executable, "CTest")

        path = shutil.which("ctest")

        if path is None:
            raise FileNotFoundError("CTest executable not found in PATH")

        return self._validate_executable(Path(path), "CTest")

    @staticmethod
    def _validate_executable(path: Path, name: str) -> Path:
        resolved_path = Path(path).resolve()

        if not resolved_path.is_file():
            raise FileNotFoundError(f"{name} executable not found: {resolved_path}")

        if not os.access(resolved_path, os.X_OK):
            raise PermissionError(
                f"{name} executable is not executable: {resolved_path}"
            )

        return resolved_path

    @staticmethod
    def _format_definition(value: str | bool | int | float | Path) -> str:
        if isinstance(value, bool):
            return "ON" if value else "OFF"

        return str(value)

    def _run(
        self,
        command: Sequence[str],
    ) -> None:
        environment = self._create_environment()

        logger.info("Running: %s", subprocess.list2cmdline(command))

        process = subprocess.Popen(
            command,
            cwd=self.build_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # Line-buffered
        )

        output_lines: list[str] = []

        assert process.stdout is not None

        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            logger.info("%s", line)

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(
                returncode=return_code,
                cmd=command,
                output="\n".join(output_lines),
            )

        pass

    def _create_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(self.environment)
        return environment


def build_cmake_project(
    source_dir: Path | str,
    build_dir: Path | str,
    *,
    build_type: str = "Release",
    target: str | None = None,
    definitions: Mapping[
        str,
        str | bool | int | float | Path,
    ]
    | None = None,
    jobs: int | None = None,
    fresh: bool = False,
) -> CMakeProject:
    """Convenience function for configuring and building a project."""

    project = CMakeProject(
        source_dir=Path(source_dir),
        build_dir=Path(build_dir),
        build_type=build_type,
        jobs=jobs,
        definitions=dict(definitions or {}),
    )

    project.configure_and_build(
        target=target,
        fresh=fresh,
    )

    return project
