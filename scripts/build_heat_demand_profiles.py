import atlite

import pandas as pd
import numpy as np

import scipy as sp


def build_heat_demand_profiles():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop_map = store['population_gridcell_map']


    cutout = atlite.Cutout('cutouts/China-2020.nc')


    #list of grid cells
    grid_cells = cutout.grid_cells()

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    hd = cutout.heat_demand(matrix=pop_matrix,
                            index=index,
                            threshold=15.,
                            a=1.,
                            constant=0.,
                            hour_shift=8.)

    with pd.HDFStore(snakemake.output.outfile, mode='w', complevel=4) as store:
        store['heat_demand_profiles'] = hd.T.to_pandas().divide(pop_map.sum())


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.input = Dict(infile="data/population/population_gridcell_map.h5")
        snakemake.output = Dict(outfile="data/heating/daily_heat_demand.h5")

    df = build_heat_demand_profiles()
