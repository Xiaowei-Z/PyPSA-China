from vresutils import Dict
import yaml

import atlite

import pandas as pd
import numpy as np

import scipy as sp


def build_solar_thermal_profiles():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop_map = store['population_gridcell_map']


    cutout = atlite.Cutout('cutouts/China-2020.nc')


    #list of grid cells
    grid_cells = cutout.grid_cells()

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    st = cutout.solar_thermal(orientation={'slope': float(snakemake.config['solar_thermal_angle']), 'azimuth': 0.},matrix=pop_matrix,index=index)

    with pd.HDFStore(snakemake.output.outfile, mode='w', complevel=4) as store:
        store['solar_thermal_profiles'] = st.T.to_pandas().divide(pop_map.sum())


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict(infile="data/population/population_gridcell_map.h5")
        snakemake.output = Dict(outfile="data/heating/solar_thermal-{angle}.h5".format(angle=snakemake.config['solar_thermal_angle']))

    build_solar_thermal_profiles()
