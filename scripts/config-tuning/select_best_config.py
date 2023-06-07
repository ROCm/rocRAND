#!/usr/bin/env -S python3

import json
import os
import re
import pandas
from pandas import DataFrame
import argparse
from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("select_best_config"),
    lstrip_blocks=True,
    trim_blocks=True
)

def guess_gcn_architecture(path: str, raw_json_data: dict):
    """
    Try to read the architecture of a rocRAND tuning benchmark JSON output file.
    """
    hdp_gcn_arch_name = raw_json_data['context']['hdp_gcn_arch_name']
    gfx_match = re.search(r'(gfx[\dA-z]+)', hdp_gcn_arch_name)
    # Or at least part of the filename
    if gfx_match:
        return gfx_match.group(0)
    return 'unknown'


def load_benchmark_result_json(path: str) -> DataFrame:
    """
    Reads and processes a JSON file produced by rocRAND's tuning benchmark.
    """
    with open(path) as f:
        raw_json_data = json.load(f)
    benchmark_data = DataFrame(raw_json_data['benchmarks'])

    gb_per_s_series = benchmark_data['bytes_per_second'] / 1024 / 1024 / 1024
    gb_per_s_series.name = 'gb_per_s'

    # Extract the groups of the regex match to a DataFrame
    name_regex = r'^(?P<generator>\S+?)_(?P<distribution>uniform|normal|log_normal|poisson)_(?P<value_type>(unsigned_)?int|short|char|long_long|float|half|double)_t(?P<block_size>\d+)_b(?P<grid_size>\d+)'
    extracted_data = benchmark_data['name'].str.extract(name_regex)

    # Merge the regex matches and the gb_per_s series
    benchmark_data = pandas.concat([extracted_data, gb_per_s_series], axis=1)

    benchmark_data['block_size_'] = benchmark_data['block_size'].str.pad(
        4, 'left', ' ')
    benchmark_data['grid_size_'] = benchmark_data['grid_size'].str.pad(
        4, 'left', ' ')

    # Add "config" as a string to handle threads and blocks together
    benchmark_data['config'] = benchmark_data['block_size'] + \
        ',' + benchmark_data['grid_size']

    # Figure out architecture
    benchmark_data['arch'] = guess_gcn_architecture(path, raw_json_data)

    return benchmark_data


def get_best_config_for_each_value_type(benchmark_data: DataFrame) -> DataFrame:
    """
    Averages the performance of the benchmarks over all benchmarked distributions.
    Selects the fastest kernel config for each value_type.
    """
    return benchmark_data.groupby(['generator', 'arch', 'value_type', 'block_size_', 'grid_size_'], as_index=False).mean(
        numeric_only=True).sort_values(['value_type', 'gb_per_s'], ascending=False).groupby(['generator', 'arch', 'value_type']).head(1)

def generate_config_files(out_dir: str, best_data: DataFrame):
    def generator_type(generator_name : str) -> str:
        if 'sobol' in generator_name.lower():
            return 'quasi'
        return 'pseudo'

    algorithm_template = env.get_template(f"config_template")
    for (key, entries) in best_data.groupby('generator'):
        default = entries[(entries.value_type == 'unsigned_int') & (entries.arch == 'gfx908')].iloc[0]
        with open(f"{out_dir}/{key}_config.hpp", "w") as text_file:
            text_file.write(algorithm_template.render(generator=key, generator_type=generator_type(key), configs=entries, config_default=default))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', nargs='+')
    parser.add_argument('--out-dir')
    args = parser.parse_args()

    benchmark = pandas.concat([load_benchmark_result_json(j) for j in args.json])
    best_configs = get_best_config_for_each_value_type(benchmark)
    if args.out_dir:
        generate_config_files(args.out_dir, best_configs)