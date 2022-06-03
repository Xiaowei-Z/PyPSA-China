# -*- coding: utf-8 -*-

import pandas as pd
from functions import pro_names

idx = pd.IndexSlice

nodes = pd.Index(pro_names)


def build_energy_totals():
    """
    list the provinces' annual space heating, hot water and electricity
    consumption.
    """

    with pd.HDFStore(snakemake.input.infile1, mode="r") as store:
        population_count = store["population"]

    with pd.HDFStore(snakemake.input.infile2, mode="r") as store:
        population_gridcell_map = store["population_gridcell_map"]

    # using DH data 2016 assuming relatively good insulation
    # MWh/hour/C/m2  source: urbanization yearbooks
    unit_space_heating = 2.4769322112272924e-06

    # In 2010, the city of Helsingborg, Sweden used 28% of its total heating 4270TJ for hot water.
    # and it has a population of 100,000
    # source: Svend DH book and wiki
    # MWh/capita/year = 4270 * 1e12 / 3.6e9 * 0.28 / 1e5
    unit_hot_water = 3.321111111111111

    # m2/capital source: urbanization yearbooks
    floor_space_per_capita = 27.28
    # 2020 27.28; 2025 32.75; 2030 36.98; 2035 40.11; 2040 42.34; 2045 43.89; 2050 44.96; 2055 45.68; 2060 46.18

    # MWh per hdh
    space_heating_per_hdd = (
        unit_space_heating * floor_space_per_capita * population_count
    )

    # MWh per day
    hot_water_per_day = unit_hot_water * population_count / 365.0

    with pd.HDFStore(snakemake.output.outfile1, mode="w", complevel=4) as store:
        store["space_heating_per_hdd"] = space_heating_per_hdd

    with pd.HDFStore(snakemake.output.outfile1, mode="a", complevel=4) as store:
        store["hot_water_per_day"] = hot_water_per_day

    return space_heating_per_hdd, hot_water_per_day


def build_co2_totals():

    iterables = [["electricity", "heating"], ["coal", "oil", "gas"]]
    midx = pd.MultiIndex.from_product(iterables, names=["sector", "fuel"])

    ## fuels for el and heating
    # source: data.stats.gov.cn (2015) unless stated otherwise
    fuels = pd.Series(index=midx)

    ## electricty
    # ton/year
    fuels["electricity", "coal"] = (165382.48 - 24095.4) * 1e4  # coal for electricity

    ## heating
    # ton/year
    # source: 中国散煤综合治理调研报告2017 http://coalcap.nrdc.cn/Public/uploads/pdf/15180772751437518672.pdf
    fuels["heating"]["coal"] = 2.34e8  # decentralized coal heating
    fuels["heating"]["coal"] += 24095.4 * 1.0e4  # DH coal
    # ton/year
    fuels["heating"]["oil"] = 493.1 * 1.0e4  # DH oil
    # m3/year
    fuels["heating"]["gas"] = 359.8 * 1.0e8  # decentralized gas heating

    ## co2 factors ton CO2 / ton (m3) fuel
    # Zhang, X., & Wang, F. (2016). Hybrid input-output analysis for life-cycle energy consumption and carbon emissions of China’s building sector. Building and Environment, 104, 188-197.
    co2_factors = pd.Series()
    co2_factors.index.name = "fuel"
    co2_factors["coal"] = 1.9901
    co2_factors["oil"] = 3.0
    co2_factors["gas"] = 21.6714 / 1.0e4

    co2 = (fuels * co2_factors).sum(level="sector")

    with pd.HDFStore(snakemake.output.outfile2, mode="w", complevel=4) as store:
        store["co2"] = co2

    return co2


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from vresutils import Dict

        snakemake = Dict()
        snakemake.input = Dict(
            infile1="data/population/population.h5",
            infile2="data/population/population_gridcell_map.h5",
        )
        snakemake.output = Dict(
            outfile1="data/energy_totals.h5", outfile2="data/co2_totals.h5"
        )

    space_heating_per_hdd, hot_water_per_day = build_energy_totals()

    co2 = build_co2_totals()
