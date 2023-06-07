#!/usr/bin/env -S python3

import json
import os
import re
import pandas
from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
from jinja2 import Environment, PackageLoader

env = Environment(
    loader=PackageLoader("select_best_config"),
    lstrip_blocks=True,
    trim_blocks=True
)


def load_benchmark_result_json(path: str) -> DataFrame:
    """
    Reads and processes a JSON file produced by rocRAND's tuning benchmark.
    """
    def guess_gcn_architecture(path: str, raw_json_data: dict):
        arch_regex = re.compile(r'gfx[0-9a-f]+')
        arch_name = raw_json_data['context']['hdp_gcn_arch_name']
        _, filename = os.path.split(path)
        for name in (arch_name, filename):
            gfx_match = re.search(arch_regex, name)
            if gfx_match:
                return gfx_match.group(0)
        return 'unknown'

    with open(path) as f:
        raw_json_data = json.load(f)
    benchmark_data = DataFrame(raw_json_data['benchmarks'])

    gb_per_s_series = benchmark_data['bytes_per_second'] / 1024 / 1024 / 1024
    gb_per_s_series.name = 'gb_per_s'

    # Extract the groups of the regex match to a DataFrame
    name_regex = r'^(?P<generator>\S+?)_(?P<distribution>uniform|normal|log_normal|poisson)_(?P<value_type>(?>unsigned_)?(?>int|short|char|long_long|float|half|double))_t(?P<block_size>\d+)_b(?P<grid_size>\d+)'
    extracted_data = benchmark_data['name'].str.extract(name_regex)

    # Merge the regex matches and the gb_per_s series
    benchmark_data = pandas.concat([extracted_data, gb_per_s_series], axis=1)

    # Figure out architecture
    benchmark_data['arch'] = guess_gcn_architecture(path, raw_json_data)
    return benchmark_data


def get_best_config_for_arch(benchmark_data: DataFrame):
    """
    Calculates the best config for each architecture, that is on average provides the highest
    performance across generated types and distributions.
    """
    def config_goodness(subf: DataFrame):
        means = mean_perf_of_configs[zip(
            subf['distribution'], subf['value_type'])]
        relative_perfs = subf['gb_per_s'].reset_index(
            drop=True) / means.reset_index(drop=True)
        return relative_perfs.sum() / relative_perfs.count()

    best_configs = {}
    for gen, generator_df in benchmark_data.groupby('generator'):
        best_config_for_generator = {}
        for arch, arch_df in generator_df.groupby('arch'):
            mean_perf_of_configs = arch_df.groupby(['distribution', 'value_type'])[
                'gb_per_s'].mean()
            config_to_goodness = arch_df.groupby(
                ['block_size', 'grid_size']).apply(config_goodness)
            best_block_size, best_grid_size = config_to_goodness.idxmax()
            best_config_for_generator[arch] = {
                'block_size': best_block_size, 'grid_size': best_grid_size}
        best_configs[gen] = best_config_for_generator
    return best_configs


def plot_benchmark_data_for_all_arches(benchmark_data: DataFrame, out_path: str):
    """
    Plots comparative figures of the benchmark results for each architecture.
    """
    def plot_benchmark_results(benchmark_data: DataFrame, arch: str, out_path: str):
        def draw_heatmap(*args, **kwargs):
            data: DataFrame = kwargs.pop('data')
            d = data.pivot_table(
                index=args[1], columns=args[0], values=args[2])
            sns.heatmap(d, **kwargs)

        grid = sns.FacetGrid(benchmark_data, col='benchmark', col_wrap=3)
        grid.map_dataframe(draw_heatmap, 'grid_size_', 'block_size_', 'gb_per_s',
                           cbar=False, square=True, annot=True, fmt='.2f', cmap='coolwarm', linewidth=1)
        fig = plt.gcf()
        fig.suptitle(f'rocRAND benchmarks ({arch}, GiB/s)')
        num_grid_sizes = benchmark_data['grid_size'].unique().astype(
            bool).sum()
        fig.set_size_inches(16/6*num_grid_sizes, 18)
        fig.savefig(out_path)

    benchmark_data['block_size_'] = benchmark_data['block_size'].str.pad(
        4, 'left', ' ')
    benchmark_data['grid_size_'] = benchmark_data['grid_size'].str.pad(
        4, 'left', ' ')
    benchmark_data['benchmark'] = benchmark_data['value_type'].astype(
        str) + ',' + benchmark_data['distribution'].astype(str)
    base_path, full_filename = os.path.split(out_path)
    filename, ext = os.path.splitext(full_filename)
    for arch in benchmark_data['arch'].unique():
        plot_benchmark_results(benchmark_data[
            benchmark_data['arch'] == arch], arch, os.path.join(base_path, f'{filename}_{arch}{ext}'))


def generate_config_files(out_dir: str, best_data: dict):
    def generator_type(generator_name: str) -> str:
        if 'sobol' in generator_name.lower():
            return 'quasi'
        return 'pseudo'

    algorithm_template = env.get_template("config_template")
    for generator, best_configs in best_data.items():
        with open(os.path.join(out_dir,  f"{generator}_config.hpp"), "w") as text_file:
            text_file.write(algorithm_template.render(
                generator=generator, generator_type=generator_type(generator), configs=best_configs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', nargs='+')
    parser.add_argument('--plot-out')
    parser.add_argument('--out-dir')
    args = parser.parse_args()

    benchmark = pandas.concat(
        [load_benchmark_result_json(j) for j in args.json])
    best_configs = get_best_config_for_arch(benchmark)
    if args.plot_out:
        plot_benchmark_data_for_all_arches(benchmark, args.plot_out)
    if args.out_dir:
        generate_config_files(args.out_dir, best_configs)
