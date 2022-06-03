# -*- coding: utf-8 -*-
import atlite
import numpy as np
import pandas as pd
import scipy as sp


def build_heat_demand_profiles():

    with pd.HDFStore(snakemake.input.infile, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    cutout = atlite.Cutout("cutouts/China-2020.nc")

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    hd = cutout.heat_demand(
        matrix=pop_matrix,
        index=index,
        threshold=15.0,
        a=1.0,
        constant=0.0,
        hour_shift=8.0,
    )

    Hd_2020 = hd.to_pandas().divide(pop_map.sum())
    Hd_2020.loc["2020-04-01":"2020-09-30"] = 0
    Hd_2020.loc[
        :,
        [
            "Anhui",
            "Sichuan",
            "Yunnan",
            "Chongqing",
            "Guizhou",
            "Guangxi",
            "Hainan",
            "Hubei",
            "Hunan",
            "Guangdong",
            "Jiangxi",
            "Shanghai",
            "Zhejiang",
            "Fujian",
        ],
    ] = 0

    with pd.HDFStore(
        snakemake.output.daily_heat_demand, mode="w", complevel=4
    ) as store:
        store["heat_demand_profiles"] = Hd_2020


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from vresutils import Dict

        snakemake = Dict()
        snakemake.input = Dict(infile="data/population/population_gridcell_map.h5")
        snakemake.output = Dict(daily_heat_demand="data/heating/daily_heat_demand.h5")

    df = build_heat_demand_profiles()
