# coding: utf-8

import logging
logger = logging.getLogger(__name__)

import pandas as pd
idx = pd.IndexSlice

import numpy as np
import xarray as xr

import pypsa
import yaml

from prepare_network import prepare_costs
from _helpers import override_component_attrs, define_spatial

from types import SimpleNamespace
spatial = SimpleNamespace()

def add_build_year_to_new_assets(n, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    baseyear : int
        year in which optimized assets are built
    """

    # Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):

        assets = c.df.index[(c.df.lifetime!=np.inf) & (c.df.build_year==0)]
        c.df.loc[assets, "build_year"] = baseyear

        # add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[assets] += "-" + str(baseyear)
        c.df.rename(index=rename, inplace=True)

        # rename time-dependent
        selection = (
            n.component_attrs[c.name].type.str.contains("series")
            & n.component_attrs[c.name].status.str.contains("Input")
        )
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr].rename(columns=rename, inplace=True)


def add_existing_capacities(df_agg):

    for tech in ['coal','CHP','solar', 'onwind', 'offwind']:

        df = pd.read_csv(snakemake.input[f"existing_{tech}"], index_col=0).fillna(0.)
        df.columns = df.columns.astype(int)
        df = df.sort_index()

        for year in df.columns:
            for node in df.index:
                name = f"{node}-{tech}-{year}"
                capacity = df.loc[node, year]
                if capacity > 0.:
                    if tech != 'CHP':
                        df_agg.at[name, "Fueltype"] = tech
                    else:
                        df_agg.at[name, "Fueltype"] = 'coal'
                    df_agg.at[name, "Capacity"] = capacity
                    df_agg.at[name, "DateIn"] = year
                    df_agg.at[name, "cluster_bus"] = node


def add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear, config):
    """
    Parameters
    ----------
    n : pypsa.Network
    grouping_years :
        intervals to group existing capacities
    costs :
        to read lifetime to estimate YearDecomissioning
    baseyear : int
    """
    print("adding power capacities installed before baseyear")

    df_agg = pd.DataFrame()

    # include renewables in df_agg
    add_existing_capacities(df_agg)

    df_agg["grouping_year"] = np.take(
        grouping_years,
        np.digitize(df_agg.DateIn, grouping_years, right=True)
    )

    df = df_agg.pivot_table(
        index=["grouping_year", 'Fueltype'],
        columns='cluster_bus',
        values='Capacity',
        aggfunc='sum'
    )

    df.fillna(0)

    for grouping_year, generator in df.index:

        # capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[capacity > snakemake.config['existing_capacities']['threshold_capacity']]

        if generator in ['coal', 'solar', 'onwind', 'offwind']:

            # to consider electricity grid connection costs or a split between
            # solar utility and rooftop as well, rather take cost assumptions
            # from existing network than from the cost database
            capital_cost = n.generators.loc[n.generators.carrier == generator, "capital_cost"].mean()

            if generator in ['solar', 'onwind', 'offwind']:
                p_max_pu = n.generators_t.p_max_pu[capacity.index + " " + generator]
                n.madd("Generator",
                       capacity.index,
                       suffix=' ' + generator + "-" + str(grouping_year),
                       bus=capacity.index,
                       carrier=generator,
                       p_nom=capacity,
                       p_nom_min=capacity,
                       p_nom_extendable=False,
                       marginal_cost=costs.at[generator, 'VOM'],
                       capital_cost=capital_cost,
                       efficiency=costs.at[generator, 'efficiency'],
                       p_max_pu=p_max_pu.rename(columns=n.generators.bus),
                       build_year=grouping_year,
                       lifetime=costs.at[generator, 'lifetime']
                       )
            else:
                p_max_pu = 1.0
                n.madd("Generator",
                       capacity.index,
                       suffix=' ' + generator + "-" + str(grouping_year),
                       bus=capacity.index,
                       carrier=generator,
                       p_nom=capacity,
                       p_nom_min=capacity,
                       p_nom_extendable=False,
                       marginal_cost=costs.at[generator, 'VOM'],
                       capital_cost=capital_cost,
                       efficiency=costs.at[generator, 'efficiency'],
                       p_max_pu=p_max_pu,
                       build_year=grouping_year,
                       lifetime=costs.at[generator, 'lifetime']
                       )
        else:
            bus0 = capacity.index + " CHP coal"

            n.madd("Link",
                   capacity.index,
                   suffix=" " + generator + "-" + str(grouping_year),
                   bus0=bus0,
                   bus1=capacity.index,
                   bus2=capacity.index + " central heat",
                   carrier=generator,
                   marginal_cost=costs.at['gas', 'fuel'],  # NB: VOM is per MWel
                   capital_cost=costs.at['central CHP', 'fixed'],  # NB: fixed cost is per MWel
                   p_nom=capacity,
                   p_nom_min=capacity,
                   p_nom_extendable=False,
                   efficiency=config['chp_parameters']['eff_el'],
                   efficiency2=config['chp_parameters']['eff_th'],
                   build_year=grouping_year,
                   lifetime=costs.at["central CHP", 'lifetime']
                   )


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'add_existing_baseyear',
            co2_reduction='0.0',
            opts ='ll',
            planning_horizons=2020
        )

    logging.basicConfig(level=snakemake.config['logging']['level'])

    options = snakemake.config["sector"]
    # sector_opts = '168H-T-H-B-I-solar+p3-dist1'
    # opts = sector_opts.split('-')

    baseyear = snakemake.config['scenario']["planning_horizons"][0]

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    # define spatial resolution of carriers
    spatial = define_spatial(n.buses[n.buses.carrier=="AC"].index, options)
    # add_build_year_to_new_assets(n, baseyear)

    Nyears = n.snapshot_weightings.generators.sum() / 8760.
    costs = prepare_costs(Nyears, snakemake.config)

    grouping_years = snakemake.config['existing_capacities']['grouping_years']
    add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear, snakemake.config)

    n.export_to_netcdf(snakemake.output[0])
