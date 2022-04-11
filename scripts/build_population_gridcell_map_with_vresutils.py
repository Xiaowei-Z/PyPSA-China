
from vresutils import shapes as vshapes, mapping as vmapping, transfer as vtransfer, load as vload

import atlite

import pandas as pd
import numpy as np

import geopandas as gpd

from functions import pro_names


def build_population_map():

    with pd.HDFStore(snakemake.input.infile, mode='r') as store:
        pop = store['population']

    pro_poly = gpd.read_file('data/province_shapes/CHN_adm1.shp')[['NAME_1', 'geometry']]
    pro_poly.replace(to_replace={'Nei Mongol': 'Inner Mongolia',
                                 'Xinjiang Uygur': 'Xinjiang',
                                 'Ningxia Hui': 'Ningxia',
                                 'Xizang':'Tibet'}, inplace=True)
    pro_poly.set_index('NAME_1', inplace=True)
    pro_poly = pro_poly.reindex(pro_names)
    pro_poly = pro_poly.geometry

    cutout = atlite.Cutout('China-2016')

    #list of grid cells
    grid_cells = cutout.grid_cells()

    #takes a few minutes
    pop_map = pd.DataFrame()

    for country in pro_names:
        print(country)
        trans_matrix = vtransfer.Shapes2Shapes(np.atleast_1d(pro_poly[country]), grid_cells)
        country_pop = pop[country]#.fillna(0.)
        pop_map[country] = np.array(trans_matrix.multiply(np.atleast_1d(country_pop)).sum(axis=1))[:,0]


    with pd.HDFStore(snakemake.output.outfile, mode='w', complevel=4) as store:
        store['population_gridcell_map'] = pop_map

if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.input = Dict(infile="data/population.h5")
        snakemake.output = Dict(outfile='data/population_gridcell_map.h5')

    build_population_map()