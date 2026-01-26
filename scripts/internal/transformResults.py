#!/usr/bin/env python

import sys
import pandas as pd
import json


def transform_cp2k(output_dir: str, df: pd.DataFrame):
    df[['type', 'width', 'internal_iterations']] = df['name'].str.extract(
        r'BM_cp2k_([^/]+)/width(\d+)/(\d+)/manual_time')

    df_compression = df[df['type'] == 'compression']
    df_decompression = df[df['type'] == 'decompression']

    # pivot the dataframes to have widths as columns
    df_compression_pivot = df_compression.pivot(index='internal_iterations', columns='width', values='bytes_per_second')
    df_decompression_pivot = df_decompression.pivot(index='internal_iterations', columns='width',
                                                    values='bytes_per_second')

    # turn internal_iterations into integer
    df_compression_pivot.index = df_compression_pivot.index.astype(int)
    df_decompression_pivot.index = df_decompression_pivot.index.astype(int)

    # sort by internal_iterations
    df_compression_pivot = df_compression_pivot.sort_index()
    df_decompression_pivot = df_decompression_pivot.sort_index()

    # save to csv in output directory
    df_compression_pivot.to_csv(f"{output_dir}/cp2k_compression.csv")
    df_decompression_pivot.to_csv(f"{output_dir}/cp2k_decompression.csv")


def transform_pernix(output_dir: str, name: str, df: pd.DataFrame):
    df[['type', 'optimization', 'memory', 'width', 'blocks']] = df['name'].str.extract(
        r'BM_([^/]+)_([^/]+)_([^/]+)_(\d+)/(\d+)')

    df_compression = df[df['type'] == 'compress']
    df_decompression = df[df['type'] == 'decompress']

    df_compression_memory_true = df_compression[df_compression['memory'] == 'true']
    df_compression_memory_false = df_compression[df_compression['memory'] == 'false']

    df_decompression_memory_true = df_decompression[df_decompression['memory'] == 'true']
    df_decompression_memory_false = df_decompression[df_decompression['memory'] == 'false']

    def pivot_and_save(df_subset: pd.DataFrame, operation: str, memory: str):
        df_pivot = df_subset.pivot(index='blocks', columns='width', values='bytes_per_second')
        df_pivot.index = df_pivot.index.astype(int)
        df_pivot = df_pivot.sort_index()
        df_pivot.to_csv(f"{output_dir}/{name}_{operation}_memory_{memory}.csv")

    pivot_and_save(df_compression_memory_true, 'compression', 'true')
    pivot_and_save(df_compression_memory_false, 'compression', 'false')
    pivot_and_save(df_decompression_memory_true, 'decompression', 'true')
    pivot_and_save(df_decompression_memory_false, 'decompression', 'false')


def transform_pernix_core_throughput(output_dir: str, name: str, df: pd.DataFrame):
    df[['type', 'optimization', 'memory', 'width', 'blocks']] = df['name'].str.extract(
        r'BM_([^/]+)_([^/]+)_([^/]+)_(\d+)/(\d+)'
    )

    # keep only memory=true, since that is what this function currently targets
    df = df[df['memory'] == 'true'].copy()

    # normalize dtypes
    df['width'] = pd.to_numeric(df['width'], errors='coerce')
    df['blocks'] = pd.to_numeric(df['blocks'], errors='coerce')
    df['bytes_per_second'] = pd.to_numeric(df['bytes_per_second'], errors='coerce')
    df = df.dropna(subset=['width', 'bytes_per_second'])

    def max_by_width(df_subset: pd.DataFrame) -> pd.DataFrame:
        # max bytes_per_second per width across all blocks
        out = (
            df_subset.groupby('width', as_index=False)['bytes_per_second']
            .max()
            .rename(columns={'width': 'bitwidth'})
            .sort_values('bitwidth')
        )

        out['bitwidth'] = out['bitwidth'].astype(int)
        return out[['bitwidth', 'bytes_per_second']]

    df_compression = df[df['type'] == 'compress']
    df_decompression = df[df['type'] == 'decompress']

    compression_out = max_by_width(df_compression)
    decompression_out = max_by_width(df_decompression)

    compression_out.to_csv(
        f"{output_dir}/{name}_core_throughput_compression_memory_true.csv",
        index=False,
    )
    decompression_out.to_csv(
        f"{output_dir}/{name}_core_throughput_decompression_memory_true.csv",
        index=False,
    )


def main():
    if len(sys.argv) != 4:
        sys.exit(1)

    name = sys.argv[1]
    input_file = sys.argv[2]
    with open(input_file, 'r') as f:
        input_data = json.load(f)["benchmarks"]
    output_directory = sys.argv[3]

    # input_data is a list of dictionaries
    df = pd.DataFrame(input_data)

    if name == "cp2k":
        transform_cp2k(output_directory, df)
    else:
        transform_pernix(output_directory, name, df)
        transform_pernix_core_throughput(output_directory, name, df)


if __name__ == '__main__':
    main()
    sys.exit(0)
