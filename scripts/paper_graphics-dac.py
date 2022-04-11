


import pandas as pd

import numpy as np

import os

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import helper

from matplotlib.patches import Circle, Ellipse

from matplotlib.legend_handler import HandlerPatch


from scipy.stats import pearsonr



def plot_time_series(flex,line_limit):

    n = helper.Network(snakemake.config['results_dir'] + "version-{}/postnetworks/postnetwork-{}-{}-0.05.nc".format(snakemake.config['version'],flex,line_limit))

    suffix = ""

    n.buses["suffix"] = n.buses.index.str[2:]

    buses = n.buses.index[n.buses.suffix == suffix]


    prices = n.buses_t.marginal_price[buses].mean(axis=1)



    co2_store = -n.stores_t.p["EU co2 Store"]/1000.

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches((5,3.5))

    df = pd.DataFrame(index=n.snapshots)

    df.index.name = None

    df["CO2 store [ktCO2/h]"] = co2_store

    df["average electricity price [EUR/MWh]"] = prices

    df.plot(alpha=0.7,ax=ax,grid=True)

    ax.set_xlim([n.snapshots[0],n.snapshots[-1]])

    ax.legend(loc="upper right")

    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/co2_v_prices.pdf".format(snakemake.config['version']),transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)

        snakemake.config["version"] = 108
        snakemake.input = Dict()
        snakemake.output = Dict()
        for item in ["costs","metrics","price_statistics","market_values","weighted_prices","supply_energy","energy","curtailment"]:
            snakemake.input[item] = snakemake.config['summary_dir'] + 'version-{version}/csvs/{item}.csv'.format(version=snakemake.config['version'],item=item)

        for item in ["co2_reduction","co2_price","price_statistics","market_values","fossil_share","curtailment"]:
            snakemake.output[item] = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/{item}.pdf'.format(version=snakemake.config['version'],item=item)


    plot_time_series("dac","opt")
