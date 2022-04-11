


import pandas as pd

import numpy as np

import os

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pypsa

from plot_summary import rename_techs, preferred_order


def plot_transmission():
    flexibilities = ["all_flex-central"]

    costs = pd.read_csv(snakemake.input.costs,header=[0,1,2,3],index_col=[0,1,2]).sort_index()
    metrics = pd.read_csv(snakemake.input.metrics,header=[0,1,2,3],index_col=[0]).sort_index()

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
        flexibility = flexibilities[i]
        df = costs[flexibility]
        df = df.rename(lambda x: metrics.loc["line_volume",(flexibility,x)]/1e6,axis=1)


        df.T.sort_index().plot(kind="area",stacked=True,linewidth=0,ax=ax,
                               color=[snakemake.config['plotting']['tech_colors'][i] for i in df.index])

        handles,labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()
        ax.set_ylim([0,800])

        ax.set_xlim([0,df.columns.to_series().max()])

        ax.set_ylabel("System Cost [EUR billion per year]")

        ax.set_xlabel("Allowed interconnecting transmission [TWkm]")

        ax.grid(axis="y")

        ax.set_title("Costs for Scenario {}".format(snakemake.config['plotting']['scenario_names'][flexibility]))

        ax.legend().set_visible(False)


        ax.plot([30,30],[0,1000],color="r",linewidth=2,linestyle="--")
        ax.plot([120,120],[0,1000],color="g",linewidth=2,linestyle="--")

        if flexibility == "base":
            today_height = 715
            compromise_height = 590
        else:
            today_height = 515
            compromise_height = 535
        ax.text(35,today_height,"today's\ngrid",size="14",color="r")
        ax.text(125,compromise_height,"compromise grid",size="14",color="g")


    #framealpha stops transparency
    #bbox: first is x, second is y
    fig.legend(handles,labels,ncol=4,bbox_to_anchor=(0.99, 0.94),framealpha=1.)#loc="upper center",
    fig.tight_layout()

    fig.savefig(snakemake.output.transmission,transparent=True)

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
        snakemake.output.transmission = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/transmission.pdf'.format(version=snakemake.config['version'])


    plot_transmission()
