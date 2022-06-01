import atlite

import pandas as pd
import numpy as np

import scipy as sp


def build_temp_profiles():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop_map = store['population_gridcell_map']

    #this one includes soil temperature
    cutout = atlite.Cutout('cutouts/China-2020.nc')

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    temp = cutout.temperature(matrix=pop_matrix,index=index)

    with pd.HDFStore(snakemake.output.temp, mode='w', complevel=4) as store:
        store['temperature'] = temp.to_pandas().divide(pop_map.sum())


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.input.infile = "data/population/population_gridcell_map.h5"
        snakemake.output = Dict()
        snakemake.output.temp = "data/heating/temp.h5"
    build_temp_profiles()
