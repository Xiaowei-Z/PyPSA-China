
from vresutils import shapes as vshapes, mapping as vmapping, transfer as vtransfer, load as vload

import atlite

import pandas as pd
import numpy as np

import scipy as sp


def build_temp_profiles():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop_map = store['population_gridcell_map']

    #this one includes soil temperature
    cutout = atlite.Cutout('China-2016')

    #list of grid cells
    grid_cells = cutout.grid_cells()

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    temp = cutout.temperature(matrix=pop_matrix,index=index)

    with pd.HDFStore(snakemake.output.outfile, mode='w', complevel=4) as store:
        store['temperature'] = temp.T.to_pandas().divide(pop_map.sum())


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.input.infile = "data/population_gridcell_map.h5"
        snakemake.output = Dict()
        snakemake.output.outfile = "data/heating/temp.h5"
    build_temp_profiles()
