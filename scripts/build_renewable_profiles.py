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

def rotate_transform(bounds, res):
    left, bottom = [(b // res) * res for b in bounds[:2]]
    right, top = [(b // res + 1) * res for b in bounds[2:]]
    return rio.Affine(res, 0, left, 0, -res, top)

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_profiles')
    configure_logging(snakemake)
    
    #add raster
    Build_up = snakemake.input['Build_up_raster'])
    Grass = snakemake.input['Grass_raster'])
    Bare= snakemake.input['Bare_raster'])
    Shrubland = snakemake.input['Shrubland_raster'])
    
    cutout = atlite.Cutout(snakemake.input['cutout'])
    
    
