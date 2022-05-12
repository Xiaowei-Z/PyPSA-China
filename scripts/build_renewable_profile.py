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
