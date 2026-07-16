# Result notebooks

This directory is reserved for reproducible exploration and visualization of
benchmark output.

Start JupyterLab from `benchmark-tools/`:

```console
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/
```

Keep reusable data loading, transformation, and plotting code in
`src/pernix_benchmark_tools/`; notebooks should focus on analysis and presentation.
