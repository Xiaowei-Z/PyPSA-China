# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

import logging
from _helpers import configure_logging

import atlite
import xarray as xr
import geopandas as gpd
from atlite.gis import shape_availability, ExclusionContainer
import numpy as np
import rasterio as rio
from shapely import ops,affinity

from functions import pro_names

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_potential')
    configure_logging(snakemake)
    
    cutout = atlite.Cutout(snakemake.input['cutout'])
    cutout.prepare()
    
    #add raster
    Build_up = snakemake.input['Build_up_raster'])
    Grass = snakemake.input['Grass_raster'])
    Bare= snakemake.input['Bare_raster'])
    Shrubland = snakemake.input['Shrubland_raster'])
    
    excluder_build_up = ExclusionContainer(crs=3035,res=500)
    excluder_build_up.add_raster(Build_up, invert=True, crs=4326)
    
    excluder = ExclusionContainer(crs=3035,res=500)
    excluder.add_raster(Grass, invert=True, crs=4326)
    excluder.add_raster(Bare, invert=True, crs=4326)
    excluder.add_raster(Shrubland, invert=True, crs=4326)
    
    country_shapes=snakemake.input['country_shapes']
    
    country_matrix = cutout.availabilitymatrix(country_shapes, excluder)
    buildup_matrix = cutout.availabilitymatrix(country_shapes, excluder_build_up)
    
    
    
    
    
    
    
    
    
