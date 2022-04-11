


import pandas as pd

import numpy as np

import os

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pypsa

from plot_summary import rename_techs, preferred_order


def plot_costs():

    line_limits = ["0","opt"]

    costs = pd.read_csv(snakemake.input.costs,header=[0,1,2],index_col=[0,1,2]).sort_index()
    metrics = pd.read_csv(snakemake.input.metrics,header=[0,1,2],index_col=[0]).sort_index()

    costs = costs.groupby(costs.index.get_level_values(2)).sum()

    costs = costs/1e9

    costs = costs.groupby(costs.index.map(rename_techs)).sum()

    to_drop = costs.index[costs.max(axis=1).fillna(0.) < snakemake.config['plotting']['costs_threshold']]

    print("dropping")

    print(costs.loc[to_drop])

    costs = costs.drop(to_drop)

    new_index = (preferred_order&costs.index).append(costs.index.difference(preferred_order))

    costs = costs.loc[new_index]

    fig, axes = plt.subplots(1,2,sharey=True)
    fig.set_size_inches((10,5))


    for i,ax in enumerate(axes):
        line_limit = line_limits[i]
        df = costs["all_flex-central"][line_limit]

        df = df.rename(lambda x: float(x),axis=1)


        df.T.sort_index().plot(kind="area",stacked=True,linewidth=0,ax=ax,
                               color=[snakemake.config['plotting']['tech_colors'][i] for i in df.index])

        handles,labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()
        ax.set_ylim([0,800])

        ax.set_xlim([0,0.5])

        ax.set_ylabel("System Cost [EUR billion per year]")

        ax.set_xlabel("CO2 emitted versus 1990")

        ax.grid(axis="y")

        ax.set_title("Costs for Transmission at {}".format(line_limit))

        ax.legend().set_visible(False)


    #framealpha stops transparency
    #bbox: first is x, second is y
    fig.legend(handles,labels,ncol=4,bbox_to_anchor=(0.99, 0.94),framealpha=1.)#loc="upper center",
    fig.tight_layout()

    fig.savefig(snakemake.output.co2_reduction,transparent=True)


def plot_price():

    line_limits = ["0","opt"]

    metrics = pd.read_csv(snakemake.input.metrics,header=[0,1,2],index_col=[0]).sort_index()

    fig, axes = plt.subplots(1,2,sharey=True)
    fig.set_size_inches((10,5))


    for i,ax in enumerate(axes):
        line_limit = line_limits[i]
        df = metrics["all_flex-central"][line_limit]

        df = df.rename(lambda x: float(x),axis=1).loc[["co2_shadow"]]


        df.T.sort_index().plot(linewidth=2,ax=ax)

        handles,labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        ax.set_ylabel("CO2 price [EUR/tCO2]")

        ax.set_xlabel("CO2 versus 1990")

        ax.set_xlim([0,0.5])

        ax.grid(axis="y")

        ax.set_title("CO2 price for Transmission {}".format(line_limit))
    fig.tight_layout()

    fig.savefig(snakemake.output.co2_price,transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.output = Dict()
        snakemake.input.costs = snakemake.config['summary_dir'] + 'version-{version}/csvs/costs.csv'.format(version=snakemake.config['version'])
        snakemake.input.metrics = snakemake.config['summary_dir'] + 'version-{version}/csvs/metrics.csv'.format(version=snakemake.config['version'])
        snakemake.output.co2_reduction = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/co2_reduction.pdf'.format(version=snakemake.config['version'])
        snakemake.output.co2_price = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/co2_price.pdf'.format(version=snakemake.config['version'])


    plot_costs()

    plot_price()
