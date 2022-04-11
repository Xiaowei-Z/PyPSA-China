# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:11:23 2016

@author: hll
"""

#import gdal
import matplotlib as mpl
mpl.use('agg')
import shapefile
import numpy as np
import networkx as nx
#from mpl_toolkits.basemap import Basemap
#from matplotlib.patches import Polygon,Wedge,Circle
#from matplotlib.colors import rgb2hex, LinearSegmentedColormap, BoundaryNorm
#import matplotlib.cm as cm
#import matplotlib.ticker as mticker
#import matplotlib.colorbar as colorbar
#from matplotlib.collections import LineCollection
#from matplotlib.lines import Line2D
#import matplotlib.transforms as mtransforms
#from matplotlib.legend_handler import HandlerPatch
import subprocess as sp
#import seaborn as sns
import pickle
import time
import datetime
import dateutil.relativedelta
import pandas as pd
import scipy.optimize
from scipy.signal import fftconvolve
from scipy import interpolate
import matplotlib.pyplot as plt
#from matplotlib.offsetbox import AnchoredText
from math import radians, cos, sin, asin, sqrt
import scipy.stats
from multiprocessing import Pool
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon, MultiPoint
#from joblib import Parallel, delayed
import geopandas as gpd
import pypsa
import shapely
import os
import warnings
from vresutils.costdata2 import get_full_cost_CO2
warnings.filterwarnings('ignore')
import io
import csv
#from docx import Document

# sns.set(font='serif', rc={'font.serif': 'Times New Roman'})

# mpl.rcdefaults()

#mpl.style.use('seaborn-ticks')
# mpl.style.use('classic')

mpl.rcdefaults()
mpl.rcParams['text.usetex'] = 'True'
mpl.rcParams['font.family'] = 'Serif'
mpl.rcParams['font.serif'] = 'Palatino'

mpl.rcParams['axes.labelsize'] = 24
#mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['figure.facecolor'] = 'white'
# mpl.rcParams['axes.prop_cycle'] = "cycler('color', ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])"
mpl.rcParams['lines.linewidth'] = '3'
# mpl.rcParams['patch.facecolor'] = '#4C72B0'
#mpl.rcParams['figure.autolayout'] = 'True'
mpl.rcParams['savefig.bbox'] = 'standard'
mpl.rcParams['legend.columnspacing'] = '0.7'
#mpl.rcParams['figure.subplot.top'] = '0.9'
mpl.rcParams['figure.figsize'] = '8, 4'
mpl.rcParams['figure.dpi'] = 72
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 8
pd.set_option('display.max_rows', 999)

CO2_const_range_num = list(0.001 * np.linspace(0, 1000, num=41))
CO2_const_range = ["%g" % x for x in CO2_const_range_num]

discountrate = get_full_cost_CO2.__globals__['discountrate']

tech_colors = {"onwind" : "b",
               'offwind' : "c",
               "hydro" : "g",
               'solar' : "y",
               "OCGT" : "brown",
               # "OCGT marginal" : "sandybrown",
               "gas" : "sandybrown",
               "transmission" : "steelblue",
               "H2" : "m",
               "battery" : "slategray",
               # "Nuclear" : "r",
               # "Nuclear marginal" : "r",
               "coal" : "k"
               }

eight_scen = ['RGDC', 'RGDC hydro', 'RGDC storage', 'RGDC hydro\&storage', 'FCG', 'FCG hydro', 'FCG storage', 'FCG hydro\&storage']

# china_colors = [sns.xkcd_rgb['golden yellow'], sns.xkcd_rgb['sky blue'], sns.xkcd_rgb['bright red'], sns.xkcd_rgb['dull orange'], sns.xkcd_rgb['grass green']]
pro_names = np.array(['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong',
       'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan',
       'Hubei', 'Hunan', 'Jiangsu', 'Jiangxi', 'Jilin', 'Liaoning',
       'InnerMongolia', 'Ningxia', 'Qinghai', 'Shaanxi', 'Shandong',
       'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang', 'Tibet',
       'Yunnan', 'Zhejiang'],
      dtype=str)

pro_names_short = np.array(['AH','BJ','CQ','FJ','GS','GD','GX','GZ','HN','HB','HLJ','HeN','HuB','HuN','JS','JX','JL','LN','NMG','NX','QH','SaX','SD','SH','SX','SC','TJ','XJ','XZ','YN','ZJ'], dtype=str)

pro_rearrange = np.array(
[1,26,9,24,18,17,16,10,23,14,30,0,3,15,22,11,12,13,5,6,8,2,25,7,29,28,21,4,20,19,27])

def sum_splitted_units(ser):
    if not ser.index.is_unique:
        ser=ser.groupby(level=(0,1)).sum()
    return ser

def regional_connections():

  r = pd.read_csv('data/edges_regions.txt', header=None)

  regional = ['-'.join([a, b]) for a, b in r.values]

  return regional

def non_regional_connections():

  r = pd.read_csv('data/edges.txt', header=None)

  all_connections = ['-'.join([a, b]) for a, b in r.values]

  regional = regional_connections()

  non_regional = [a for a in all_connections if a not in regional ]

  return non_regional


##########################
##some value assignments##
##########################


# return rate
r = 0.04

# cost table
costass = np.array([[6.6, 33, 410, 30],[8.2, 117, 0, 25],[8.6, 220, 0, 25],[14.7, 660, 0, 30]])

costass_10 = np.array([[6.6, 33, 410, 30],[7.38, 117, 0, 25],[8.6, 220, 0, 25],[14.7, 660, 0, 30]])

costass_25 = np.array([[6.6, 33, 410, 30],[6.15, 117, 0, 25],[8.6, 220, 0, 25],[14.7, 660, 0, 30]])

costass_50 = np.array([[6.6, 33, 410, 30],[4.1, 117, 0, 25],[8.6, 220, 0, 25],[14.7, 660, 0, 30]])

# CF calculated using RE-atlas 2005-2012 weather years and best 40% locations in each province
CFw = np.array([ 0.1761411 ,  0.17279604,  0.10488644,  0.16099106,  0.24973537,
    0.19104962,  0.17145643,  0.13769612,  0.31188691,  0.28984153,
    0.23998563,  0.18132613,  0.15583397,  0.14234001,  0.32883186,
    0.13073918,  0.26143027,  0.30031943,  0.35091275,  0.26406171,
    0.28892473,  0.21634999,  0.29206475,  0.45654259,  0.19720379,
    0.23053681,  0.23004931,  0.2465653 ,  0.30040212,  0.15958981,
    0.17128541])

CFs = np.array([ 0.16005944,  0.19181205,  0.1538134 ,  0.13532975,  0.21276674,
    0.14188135,  0.13601111,  0.12596166,  0.15690507,  0.19394156,
    0.18284049,  0.17669632,  0.16275363,  0.13729327,  0.1668826 ,
    0.13786809,  0.18900758,  0.19115015,  0.20642439,  0.20313343,
    0.24618224,  0.19296075,  0.18058521,  0.15229278,  0.19074925,
    0.21763786,  0.18609533,  0.22359826,  0.26173796,  0.15787896,
    0.14020787])

# from sparse_HCF.npy and onnearshoremap.npy & heights.npy
onshoreshare = np.array([ 1.        ,  1.        ,  1.        ,  0.90909091,  1.        ,
    0.81818182,  0.97752809,  1.        ,  0.47058824,  0.85714286,
    1.,  1.        ,  1.        ,  1.        ,  0.37037037,
    1.        ,  1.        ,  0.82352941,  1.        ,  1.        ,
    1.        ,  1.        ,  0.51282051,  0.        ,  1.        ,
    1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
    0.86842105])

# transmission line cost per km for each province
Pl = np.array([ 2000.,  2000.,  2000.,  2000.,  5000.,  5000.,  5000.,  2000.,
        2000.,  2000.,  5000.,  5000.,  2000.,  2000.,  5000.,  5000.,
        2000.,  5000.,  2000.,  2000.,  2000.,  2000.,  2000.,  5000.,
        5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  2000.,  2000.,
        5000.,  2000.,  5000.,  2000.,  2000.,  5000.,  2000.,  2000.,
        2000.,  5000.,  5000.,  5000.,  2000.,  2000.,  2000.,  2000.,
        2000.,  5000.])

# length of tramission lines in km
dl = np.array([ 414,  473,  283,  400,  378,  121,  114,  529,  447,  458,  431,
    352,  387, 1382,  535,  575,  452,  670,  754,  636,  501,  390,
    508,  726,  335,  222,  713,  409, 1264,  426,  437,  452,  331,
    477,  393,  433,  435,  427,  380,  268,  436,  276, 1040,  910,
   1029,  583,  399,  515,  233, 1299])

##########################
##########################
##########################


def draw_text(ax, text, loc, alpha):
    #at = AnchoredText(text, loc=loc, frameon=True, prop=dict(size=12), pad=0.08, borderpad=0.7)
    at = AnchoredText(text, loc=loc, frameon=True, pad=0.08, borderpad=0.7)
    at.patch.set_boxstyle('round, pad=0.3')
    at.patch.set_facecolor('white')
    at.patch.set_edgecolor('black')
    at.patch.set_alpha(alpha)
    ax.add_artist(at)



def rearrange_provinces(old_array):

    return old_array[pro_rearrange.argsort()]


def layout_for_wind_conversion(year000, extra_effi):

    for province in range(31):

        year = year000 - 2005

#        turbine_cap = 1.5
        turbine_cap = 2.3

        all_masks = np.load('/home/hll/china/sparse_HCF.npy')

#        wind_cap = np.load('/home/hll/china/wind_cap.npy')
        wind_cap = np.ones((31, 11))

        pro_mask = all_masks[province, :, :]

#==============================================================================
#         if year>0:
#             pro_cap = (wind_cap[province, year] + wind_cap[province, year-1]) / 2
#         else:
#             pro_cap = wind_cap[province, year]
#==============================================================================

        pro_cap = wind_cap[province, year]

        pro_layout = extra_effi[province] * pro_cap / np.sum(pro_mask) / turbine_cap * pro_mask

        np.save('/home/hll/REatlas-client/wind_layouts/layout_' + str(province) + '.npy', pro_layout)


def layout_for_solar_conversion(year000, extra_effi):

    for province in range(31):

        year = year000 - 2005

        PV_cap = 156 # W

        all_masks = np.load('/home/hll/china/sparse_HS.npy')

#        solar_cap = np.load('/home/hll/china/solar_cap.npy')
        solar_cap = np.ones((31, 11))

        solar_cap = solar_cap * 1000000 # from MW to W

        pro_mask = all_masks[province, :, :]

#==============================================================================
#         if year>0:
#             pro_cap = (solar_cap[province, year] + solar_cap[province, year-1]) / 2
#         else:
#             pro_cap = solar_cap[province, year]
#==============================================================================

        pro_cap = solar_cap[province, year]

        pro_layout = extra_effi[province] * pro_cap / np.sum(pro_mask) / PV_cap * pro_mask

        np.save('/home/hll/REatlas-client/solar_layouts/layout_' + str(province) + '.npy', pro_layout)


def plot_pro_weights(weights, cmap):

    sf = shapefile.Reader('CHN_adm1')
    shapes = sf.shapes()

    m = Basemap(
            projection='merc', llcrnrlon=70, llcrnrlat=15,
            urcrnrlon=140, urcrnrlat=55, lat_0=15, lon_0=95, resolution='c')

    m.readshapefile('CHN_adm1', 'pro_shp')
    m.readshapefile('TWN_adm0', 'twn_shp')

#==============================================================================
#     weights = np.load('wind_cap.npy')[:, 9]
#==============================================================================

    weights_norm = [float(i)/max(weights) for i in weights]

#==============================================================================
#      redgreendict = {'red': [(0.0, 0.8, 0.8), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
#                      'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.7, 0.7)],
#                      'blue': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]}
#      cmap = LinearSegmentedColormap('redgreen', redgreendict, 1000)
#==============================================================================



    ax = plt.gca()

    for pro_sq in range(31):
        lons, lats = zip(*shapes[pro_sq].points)
        data = np.array(m(lons, lats)).T
        if len(shapes[pro_sq].parts) == 1:
                segs = [data, ]
        else:
            segs = []
            for i in range(1, len(shapes[pro_sq].parts)):
                index = shapes[pro_sq].parts[i - 1]
                index2 = shapes[pro_sq].parts[i]
                segs.append(data[index:index2])
            segs.append(data[index2:])

        lines = LineCollection(segs, antialiaseds=(1,))
        color = cmap(weights_norm[pro_sq])
        lines.set_facecolor(color)
        lines.set_edgecolors('k')
        lines.set_linewidth(0.3)
        ax.add_collection(lines)

def submit_solar_conversion(year, conversion_name, extra_effi):

    layout_for_solar_conversion(year, extra_effi)

    cmdRE='python /home/hll/REatlas-client/cmd_convert_and_aggregate_PV.py --username hailiang --password hailiang pepsimax.imf.au.dk China_' + str(year) + ' /home/hll/REatlas-client/SolarPanelData/Scheuten215IG.cfg  /home/hll/REatlas-client/orientation_examples/latitude_optimal.cfg' + ' /home/hll/REatlas-client/solar_layouts/layout_0.npy /home/hll/REatlas-client/solar_layouts/layout_1.npy /home/hll/REatlas-client/solar_layouts/layout_2.npy /home/hll/REatlas-client/solar_layouts/layout_3.npy /home/hll/REatlas-client/solar_layouts/layout_4.npy /home/hll/REatlas-client/solar_layouts/layout_5.npy /home/hll/REatlas-client/solar_layouts/layout_6.npy /home/hll/REatlas-client/solar_layouts/layout_7.npy /home/hll/REatlas-client/solar_layouts/layout_8.npy /home/hll/REatlas-client/solar_layouts/layout_9.npy /home/hll/REatlas-client/solar_layouts/layout_10.npy /home/hll/REatlas-client/solar_layouts/layout_11.npy /home/hll/REatlas-client/solar_layouts/layout_12.npy /home/hll/REatlas-client/solar_layouts/layout_13.npy /home/hll/REatlas-client/solar_layouts/layout_14.npy /home/hll/REatlas-client/solar_layouts/layout_15.npy /home/hll/REatlas-client/solar_layouts/layout_16.npy /home/hll/REatlas-client/solar_layouts/layout_17.npy /home/hll/REatlas-client/solar_layouts/layout_18.npy /home/hll/REatlas-client/solar_layouts/layout_19.npy /home/hll/REatlas-client/solar_layouts/layout_20.npy /home/hll/REatlas-client/solar_layouts/layout_21.npy /home/hll/REatlas-client/solar_layouts/layout_22.npy /home/hll/REatlas-client/solar_layouts/layout_23.npy /home/hll/REatlas-client/solar_layouts/layout_24.npy /home/hll/REatlas-client/solar_layouts/layout_25.npy /home/hll/REatlas-client/solar_layouts/layout_26.npy /home/hll/REatlas-client/solar_layouts/layout_27.npy /home/hll/REatlas-client/solar_layouts/layout_28.npy /home/hll/REatlas-client/solar_layouts/layout_29.npy /home/hll/REatlas-client/solar_layouts/layout_30.npy' + ' --name ' + str(conversion_name)

    p = sp.Popen(cmdRE, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    stringwithjobid = stdout.splitlines()[6].split()

    jobID = [int(s) for s in stringwithjobid if s.isdigit()][0]

    resultName = stdout.splitlines()[7].split()[3][:-1].decode("utf-8")

    return jobID, resultName

def submit_wind_conversion(year, conversion_name, extra_effi):

    layout_for_wind_conversion(year, extra_effi)

    cmdRE='python /home/hll/REatlas-client/cmd_convert_and_aggregate_wind.py pepsimax.imf.au.dk China_' + str(year) + '  /home/hll/REatlas-client/TurbineConfig/smoothed/Siemens_SWT_2300kW.cfg /home/hll/REatlas-client/TurbineConfig/smoothed/Siemens_SWT_2300kW.cfg /home/hll/REatlas-client/wind_layouts/layout_0.npy  /home/hll/REatlas-client/wind_layouts/layout_1.npy /home/hll/REatlas-client/wind_layouts/layout_2.npy /home/hll/REatlas-client/wind_layouts/layout_3.npy /home/hll/REatlas-client/wind_layouts/layout_4.npy /home/hll/REatlas-client/wind_layouts/layout_5.npy /home/hll/REatlas-client/wind_layouts/layout_6.npy /home/hll/REatlas-client/wind_layouts/layout_7.npy /home/hll/REatlas-client/wind_layouts/layout_8.npy /home/hll/REatlas-client/wind_layouts/layout_9.npy /home/hll/REatlas-client/wind_layouts/layout_10.npy /home/hll/REatlas-client/wind_layouts/layout_11.npy /home/hll/REatlas-client/wind_layouts/layout_12.npy /home/hll/REatlas-client/wind_layouts/layout_13.npy /home/hll/REatlas-client/wind_layouts/layout_14.npy /home/hll/REatlas-client/wind_layouts/layout_15.npy /home/hll/REatlas-client/wind_layouts/layout_16.npy /home/hll/REatlas-client/wind_layouts/layout_17.npy /home/hll/REatlas-client/wind_layouts/layout_18.npy /home/hll/REatlas-client/wind_layouts/layout_19.npy /home/hll/REatlas-client/wind_layouts/layout_20.npy /home/hll/REatlas-client/wind_layouts/layout_21.npy /home/hll/REatlas-client/wind_layouts/layout_22.npy /home/hll/REatlas-client/wind_layouts/layout_23.npy /home/hll/REatlas-client/wind_layouts/layout_24.npy /home/hll/REatlas-client/wind_layouts/layout_25.npy /home/hll/REatlas-client/wind_layouts/layout_26.npy /home/hll/REatlas-client/wind_layouts/layout_27.npy /home/hll/REatlas-client/wind_layouts/layout_28.npy /home/hll/REatlas-client/wind_layouts/layout_29.npy /home/hll/REatlas-client/wind_layouts/layout_30.npy --username hailiang --password hailiang'  + ' --name ' + str(conversion_name)

    p = sp.Popen(cmdRE, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    jobID = [int(s) for s in stdout.splitlines()[6].split() if s.isdigit()][0]

    resultName = stdout.splitlines()[7].split()[3][:-1].decode("utf-8")

    return jobID, resultName

def download_conversion_result_wind(year, jobID, resultName):

    cmdRE_wind = 'python /home/hll/REatlas-client/cmd_get_result.py pepsimax.imf.au.dk ' + str(jobID) + ' ' + resultName + ' China_wind_' + str(year) + '.npy --username hailiang --password hailiang'

    sp.run([cmdRE_wind], shell=True)

def download_conversion_result_solar(year, jobID, resultName):

    cmdRE_solar = 'python /home/hll/REatlas-client/cmd_get_result.py pepsimax.imf.au.dk ' + str(jobID) + ' ' + resultName + ' China_solar_' + str(year) + '.npy --username hailiang --password hailiang'

    sp.run([cmdRE_solar], shell=True)

def onnearshoremap(depth):

    heights = np.load('/home/hll/REatlas-client/China_2014_metadata_hll/heights.npy')

    onnearshore = heights>-depth

    onshore = np.load('/home/hll/REatlas-client/China_2014_metadata_hll/onshoremap.npy')

    onnearshore = onshore + onnearshore

    onnearshore = onnearshore>0

    np.save('/home/hll/REatlas-client/China_2014_metadata_hll/onnearshore.npy', onnearshore)



def infrastructure(alphas, q, gamma, wind, solar, load_ts):

    cap_factor_wind = np.mean(wind, axis=0)
    cap_factor_solar = np.mean(solar, axis=0)

    mload = np.mean(load_ts, axis=0) # MWh
    ann_load_total = np.sum(load_ts, axis=0) # MWh

    E_B =[]
    K_B=[]
    K_wind=[]
    K_solar=[]

    for alpha in alphas:

        alpha = alpha * np.ones(31)

        cap_wind = alpha * gamma * mload / cap_factor_wind
        cap_solar = (1-alpha) * gamma * mload / cap_factor_solar

        wind_ts = np.multiply(cap_wind, wind)
        solar_ts = np.multiply(cap_solar, solar)

        mismatch = wind_ts + solar_ts - load_ts
#        var_mis =

        balancing = mismatch # for zero-tranmission
        backup_ts = -np.fmin(balancing, 0)
        backup_gen = np.nansum(backup_ts, axis=0)
#        backup_energy = np.sum(backup_gen) / np.sum(ann_load)
        backup_energy = backup_gen / ann_load_total

        backup_quantile = np.percentile(backup_ts, q, axis=0)
#        backup_capacity = np.sum(backup_quantile) / np.sum(mload)
        backup_capacity = backup_quantile / mload

        E_B.append(backup_energy)
        K_B.append(backup_capacity)
        K_wind.append(cap_wind)
        K_solar.append(cap_solar)

    E_B = np.array(E_B)
    K_B = np.array(K_B)
    K_wind = np.array(K_wind)
    K_solar = np.array(K_solar)

    return E_B, K_B, K_wind, K_solar

def cal_LCOE(E_B, K_B, K_wind, K_solar, Kl_T):
    costass = np.array([[6.6, 33, 410, 30],[13, 117, 0, 25],[8.8, 220, 0, 25],[17.6, 660, 0, 30]])

    # dummy numbers
    # E_B = np.ones(31) # MWh/year
    # K_B = np.ones(31) # MW
    # K_solar = np.ones(31) #MW
    # K_wind = np.ones(31) #MW
    r = 0.04 # return

    # Backup capacity investment
    v_BK = K_B * 1000000 * costass[0, 0] + np.sum((K_B * 1000 * costass[0, 1]) / (1 + r) ** t for t in range(int(costass[0, 3])))

    # Backup energy investment
    v_BE = np.sum((E_B * costass[0, 2]) / (1 + r) ** t for t in range(int(costass[0, 3])))

    # PV investment
    v_S = K_solar * 1000000 * costass[1, 0] + np.sum(K_solar * 1000 * costass[1, 1] / (1 + r) ** t for t in range(int(costass[1, 3])))

    # Wind onshore investment
    sparse_HCF = np.load('sparse_HCF.npy')
    onshoremap = np.load('/home/hll/REatlas-client/China_2014_metadata_hll/onshoremap.npy')
    onshoreshare = np.sum(np.multiply(sparse_HCF, onshoremap), axis=(1,2)) / np.sum(sparse_HCF, axis=(1,2))

    K_wind_onshore = K_wind * onshoreshare

    v_Won = K_wind_onshore * 1000000 * costass[2, 0] + np.sum(K_wind_onshore * 1000 * costass[2, 1] / (1 + r) ** t for t in range(int(costass[2, 3])))

    # Wind offshore investment

    K_wind_offshore = K_wind * (1 - onshoreshare)

    v_Woff = K_wind_offshore * 1000000 * costass[3, 0] + np.sum(K_wind_offshore * 1000 * costass[3, 1] / (1 + r) ** t for t in range(int(costass[3, 3])))

    # Transmission investment
    # dummy number
    Pl = np.load('/home/hll/pylibs/settings/edge_expense_per_km.npy')
    dl = np.load('/home/hll/pylibs/settings/length_km.npy')
    v_Tran = np.sum(Kl_T * dl * Pl) + sum(1 for line in Pl if line==7.5*1500) * 150000 * 7.5

    load_ts = np.load('load_time_series.npy') * 1000000 # MWh over 10 years
    load_pro_year = np.sum(load_ts, axis=0) / 10 # MWh/year for 31 provinces
    load_year = np.sum(load_ts, axis=(0,1)) / 10 # MWh/year for China

    ### LCOE per province -- only suitable for zero-transmission
    #LCOE_B = v_B / (np.sum(load_pro_year / (1 + r) ** t for t in range(int(costass[0, 3]))))
    #LCOE_S = v_S / (np.sum(load_pro_year / (1 + r) ** t for t in range(int(costass[1, 3]))))
    #LCOE_Won = v_Won / (np.sum(load_pro_year / (1 + r) ** t for t in range(int(costass[2, 3]))))
    #LCOE_Woff = v_Woff / (np.sum(load_pro_year / (1 + r) ** t for t in range(int(costass[3, 3]))))
    #
    #LCOE = LCOE_B + LCOE_S + LCOE_Won + LCOE_Woff

    ## LCOE for China as a whole
    LCOE_BK = np.sum(v_BK) / (np.sum(load_year / (1 + r) ** t for t in range(int(costass[0, 3]))))
    LCOE_BE = np.sum(v_BE) / (np.sum(load_year / (1 + r) ** t for t in range(int(costass[0, 3]))))
    LCOE_S = np.sum(v_S) / (np.sum(load_year / (1 + r) ** t for t in range(int(costass[1, 3]))))
    LCOE_Won = np.sum(v_Won) / (np.sum(load_year / (1 + r) ** t for t in range(int(costass[2, 3]))))
    LCOE_Woff = np.sum(v_Woff) / (np.sum(load_year / (1 + r) ** t for t in range(int(costass[3, 3]))))
    LCOE_T = np.sum(v_Tran) / (np.sum(load_year / (1 + r) ** t for t in range(int(40))))

    LCOE = LCOE_BK + LCOE_BE + LCOE_S + LCOE_Won + LCOE_Woff + LCOE_T

    return [LCOE_S, LCOE_Won+LCOE_Woff, LCOE_BK, LCOE_BE, LCOE_T]

def distance_two_points_on_earth(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def China_network():
    pro_loca = np.load('pro_loca.npy')

    G = nx.Graph()
    G.add_nodes_from(range(31))
    G.add_edges_from(np.load('edges.npy'))

    # length_km
    for n1, n2 in nx.edges_iter(G):
        distance = distance_two_points_on_earth(pro_loca[n1, 0], pro_loca[n1, 1], pro_loca[n2, 0], pro_loca[n2, 1])
        G.edge[n1][n2]['length_km'] = int(round(distance))

    # HVDC
    for n1, n2 in nx.edges_iter(G):
        if (n1, n2) in [(0, 11), (0, 12), (0, 14), (0, 30), (2, 21), (2, 25), (2, 12), (4, 27), (4, 19), (5, 6), (6, 29), (7, 13), (7, 25), (9, 18), (9, 22), (11, 22), (11, 24), (12, 15), (12, 21), (13, 15), (14, 22), (14, 23), (15, 30), (18, 24), (18, 21), (19, 21), (21, 24), (23, 30)]:
            G.edge[n1][n2]['HVDC'] = 1
        else:
            G.edge[n1][n2]['HVDC'] = 0

    return G


def haversine(p1,p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)"""

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    n = float(n)
    r = float(r)

    if r == 0:
        return 1/n
    else:
        return r/(1. -1./(1.+r)**n)


override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]

def network_for_plot_func(csv_folder_name):

    network = pypsa.Network(csv_folder_name, override_component_attrs=override_component_attrs)

    network.options = pd.read_csv(os.path.join(csv_folder_name,'options.csv'))

    network_for_plot = network.copy()

    for component in network_for_plot.buses.index[31:]:
        network_for_plot.remove('Bus', component)

    for component in network_for_plot.generators[network_for_plot.generators.carrier=='hydro_inflow'].index:
        network_for_plot.remove('Generator', component)

    return network_for_plot

def network_for_plot_netcdf_func(csv_folder_name):

    network = pypsa.Network(csv_folder_name+'/netcdf.nc', override_component_attrs=override_component_attrs)

    network.options = pd.read_csv(os.path.join(csv_folder_name,'options.csv'))

    network_for_plot = network.copy()

    for component in network_for_plot.buses.index[31:]:
        network_for_plot.remove('Bus', component)

    for component in network_for_plot.generators[network_for_plot.generators.carrier=='hydro_inflow'].index:

        network_for_plot.remove('Generator', component)

    return network_for_plot

# def multi_const_networks_costs_func(tech_str, LV, net_topo):

#     multi_const_networks_dict = multi_const_networks_func(tech_str, LV, net_topo)

#     multi_const_networks_costs = dict()
#     transmission_costs = dict()

#     for CO2_const in CO2_const_range:

#         the_network = multi_const_networks_dict[CO2_const]

#         multi_const_networks_costs[CO2_const] = calc_costs_countries(the_network)

#         for link in the_network.links.index:

#             if the_network.links.p_min_pu[link]==-1:
#                 # AC lines
#                 the_network.links.at[link, "capital_cost"] = (1. * the_network.links.at[link, "length"] * CVAC_cost_curve(the_network.links.at[link, "length"]) * 1.25) * 1.5 * 1.02 * the_network.snapshot_weightings.sum()/8760.*annuity(40.,discountrate)

#             elif the_network.links.p_min_pu[link]==0:
#                 # Unidirectional DC lines
#                 the_network.links.at[link, "capital_cost"] = (1. * the_network.links.at[link, "length"] * HVDC_cost_curve(the_network.links.at[link, "length"]) * 1.25 + 150000.) * 1.5 * 1.02 * the_network.snapshot_weightings.sum()/8760.*annuity(40.,discountrate)

#         transmission_costs[CO2_const] = sum(the_network.links.at[link,"capital_cost"] * the_network.links.at[link,"p_nom_opt"] for link in the_network.links[the_network.links.p_min_pu==-1].index) / the_network.loads_t.p_set.sum().sum()

#     return multi_const_networks_costs, transmission_costs


def transmission_costs_func(the_network):

    for link in the_network.links.index:

        if (the_network.links.p_min_pu[link]==-1) and (link in regional_connections()):
            # AC lines
            the_network.links.at[link, "capital_cost"] = (1. * the_network.links.at[link, "length"] * CVAC_cost_curve(the_network.links.at[link, "length"]) * 1.25) * 1.5 * 1.02 * the_network.snapshot_weightings.sum()/8760.*annuity(40.,discountrate)

        elif (the_network.links.p_min_pu[link]==-1) and (link in non_regional_connections()):
            # AC lines
            the_network.links.at[link, "capital_cost"] = (1. * the_network.links.at[link, "length"] * HVAC_cost_curve(the_network.links.at[link, "length"]) * 1.25) * 1.5 * 1.02 * the_network.snapshot_weightings.sum()/8760.*annuity(40.,discountrate)

        elif (the_network.links.p_min_pu[link]==0) and ('turbines' not in link) and ('spillage' not in link):
            # Unidirectional DC lines
            the_network.links.at[link, "capital_cost"] = (1. * the_network.links.at[link, "length"] * HVDC_cost_curve(the_network.links.at[link, "length"]) * 1.25 + 150000.) * 1.5 * 1.02 * the_network.snapshot_weightings.sum()/8760.*annuity(40.,discountrate)

        transmission_costs = sum(the_network.links.at[link,"capital_cost"] * the_network.links.at[link,"p_nom_opt"] for link in the_network.links[the_network.links.p_min_pu==-1].index) / the_network.loads_t.p_set.sum().sum()

        return transmission_costs


# def multi_const_networks_func(tech_str, LV, net_topo):

#     multi_const_networks_dict = dict()

#     for CR in CO2_const_range:

#         csv_folder_name = 'results/hll/diw2030-CarbonPrice_0.0-2016_2017-'+tech_str+'-CO2_reduction_'+CR+'-Transmission_limit_factor_'+LV+net_topo

#         multi_const_networks_dict[CR] = network_for_plot_netcdf_func(csv_folder_name)

#     return multi_const_networks_dict


def tech_costs_sum(tech_str, LV, net_topo):

    costs = pd.DataFrame()
    co2_prices = pd.DataFrame()

    # multi_const_networks_costs, transmission_costs = multi_const_networks_costs_func(tech_str, LV, net_topo)

    for count, CO2_const in enumerate(CO2_const_range):

        csv_folder_name = 'results/hll/diw2030-CarbonPrice_0.0-2016_2017-'+tech_str+'-CO2_reduction_'+CO2_const+'-Transmission_limit_factor_'+LV+net_topo

        try:

          network = network_for_plot_netcdf_func(csv_folder_name)

        except IOError:

          break

        costs.at[:, CO2_const] = calc_costs_countries(network).groupby(level=[1]).sum()

        costs.at['transmission', CO2_const] = transmission_costs_func(network)

        co2_prices.at['co2_prices', CO2_const] = network.global_constraints.mu.values[0]

    tech_str_dict = dict(zip(['w','W','s','g','r','H','b','c'], ['onwind','offwind','solar','gas','hydro','H2','battery','coal']))

    tech_str_list = [tech_str_dict[v] for v in list(tech_str)] + ['transmission']

    tech_order = ['coal', 'OCGT', 'gas', 'hydro', 'offwind', 'onwind', 'solar', 'H2', 'battery', 'transmission']

    intersection=[i for i in tech_order if i in tech_str_list]

    intersection.insert(intersection.index('gas'), 'OCGT')

    costs = costs.T[intersection]

    costs.index = CO2_const_range_num[:len(costs.index)]

    costs.index.name = '$CO_2$ emission reduction'

    return costs, co2_prices

def HVDC_cost_curve(distance):

  d = np.array([1668, 1891, 2059, 2208])
  c = 1000 / 7.5 * np.array([1.75, 1.57, 1.49, 1.27])

  c_func = interpolate.interp1d(d, c, fill_value='extrapolate')
  c_results = c_func(distance)

  return c_results

def CVAC_cost_curve(distance):

  d = np.array([270, 430])
  c = 1000 / 7.5 * np.array([1.2, 2.])

  c_func = interpolate.interp1d(d, c, fill_value='extrapolate')
  c_results = c_func(distance)

  return c_results

def HVAC_cost_curve(distance):

  d = np.array([608, 656, 730, 780, 903, 920, 1300])
  c = 1000 / 7.5 * np.array([5.5, 4.71, 5.5, 5.57, 5.5, 5.5, 5.51])

  c_func = interpolate.interp1d(d, c, fill_value='extrapolate')
  c_results = c_func(distance)

  return c_results

def calc_costs_countries(network):


  gens = network.generators.set_index(['bus','carrier'])

  generator_capital = gens.capital_cost * gens.p_nom_opt
  gen_p = network.generators_t.p.sum()
  new_index = np.array([[" ".join(generator_name.split()[:-1]), generator_name.split()[-1]] for generator_name in network.generators_t.p.columns])
  gen_p = gen_p.to_frame()
  gen_p.set_index([new_index[:, 0], new_index[:, 1]], append=False, inplace=True)
  gen_p.index.names = ['bus', 'carrier']
  gen_p = gen_p.iloc[:,0]

  generator_marginal = gen_p * gens.marginal_cost

  generator_marginal.index = generator_marginal.index.set_levels(generator_marginal.index.levels[1].str.replace('OCGT', 'gas'), level=1)

  sus = network.storage_units.set_index(['bus','carrier'])

  storage_capital = sus.capital_cost * sus.p_nom_opt

  costs_countries = pd.concat([sum_splitted_units(ser) for ser in [generator_capital,generator_marginal,storage_capital]],
                              axis=1,join='outer').sum(axis=1)

  costs_countries = costs_countries.iloc[costs_countries.index.get_level_values(1)!='generation']
  costs_countries.index = costs_countries.index.remove_unused_levels()

  costs_countries_average = costs_countries / network.loads_t.p_set.sum().sum()

  return costs_countries_average


def calc_energy_countries(network):

  gen_p = network.generators_t.p.sum()
  new_index = np.array([[" ".join(generator_name.split()[:-1]), generator_name.split()[-1]] for generator_name in network.generators_t.p.columns])
  gen_p = gen_p.to_frame()
  gen_p.set_index([new_index[:, 0], new_index[:, 1]], append=False, inplace=True)
  gen_p.index.names = ['bus', 'carrier']
  gen_p = gen_p.iloc[:,0]

  c = pd.concat([sum_splitted_units(ser) for ser in [gen_p]],
                              axis=1,join='outer').sum(axis=1)

  c = c.iloc[(c.index.get_level_values(1)!='generation') & (c.index.get_level_values(1)!='inflow')]
  c.index = c.index.remove_unused_levels()

  df = network.links_t.p1.filter(like='turbines').rename(index=str, columns=network.links.bus1.filter(like='turbines').to_dict())

  h = - df.groupby(df.columns, axis=1).sum().sum()

  h.index = [h.index, len(h.index)*['hydro']]

  h.index.names = ['bus', 'carrier']

  energy_countries = pd.concat([c,h]).groupby(level=(0,1)).sum()

  return energy_countries


def load_network(year, hydro_bool, CO2_reduction, LV):

    override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
    override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
    override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
    override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]

    csv_folder_name = 'results/hll/diw2030-CarbonPrice_0.0-' +str(year)+'_'+str(year+1)+'-wWsg' +('r' if hydro_bool else '')+ 'Hb-CO2_reduction_'+str(CO2_reduction)+'-Transmission_limit_factor_'+str(LV)

    network = pypsa.Network(csv_folder_name, override_component_attrs=override_component_attrs)

    network.options = pd.read_csv(os.path.join(csv_folder_name,'options.csv'))

    network_for_plot = network.copy()

    for component in network_for_plot.buses.index[31:]:
        network_for_plot.remove('Bus', component)

    for component in network_for_plot.generators[network_for_plot.generators.carrier=='hydro_inflow'].index:
        network_for_plot.remove('Generator', component)


    return network_for_plot



def read_docx_tables(filename, tab_id=None, **kwargs):
    """
    parse table(s) from a Word Document (.docx) into Pandas DataFrame(s)

    Parameters:
        filename:   file name of a Word Document

        tab_id:     parse a single table with the index: [tab_id] (counting from 0).
                    When [None] - return a list of DataFrames (parse all tables)

        kwargs:     arguments to pass to `pd.read_csv()` function

    Return: a single DataFrame if tab_id != None or a list of DataFrames otherwise
    """
    def read_docx_tab(tab, **kwargs):
        vf = io.StringIO()
        writer = csv.writer(vf)
        for row in tab.rows:
            writer.writerow(cell.text for cell in row.cells)
        vf.seek(0)
        return pd.read_csv(vf, **kwargs)

    doc = Document(filename)
    if tab_id is None:
        return [read_docx_tab(tab, **kwargs) for tab in doc.tables]
    else:
        try:
            return read_docx_tab(doc.tables[tab_id], **kwargs)
        except IndexError:
            print('Error: specified [tab_id]: {}  does not exist.'.format(tab_id))
            raise

def wind_solar_generators_func(network):

    return [string for string in list(network.generators.index) if any([a in string for a in ['wind', 'solar']]) ]


def hydro_generators_func(network):

    return [string for string in list(network.links_t.p1.columns) if any([a in string for a in ['turbines']]) ]

def wind_solar_curtailment_rate_func(network):

    wind_solar_generators = wind_solar_generators_func(network)

    wind_solar_generation_max = network.generators_t.p_max_pu.loc[:,wind_solar_generators] * network.generators.p_nom_opt.loc[wind_solar_generators]

    wind_solar_generation = network.generators_t.p.loc[:,wind_solar_generators]

    wind_solar_curtailment_rate = (wind_solar_generation_max - wind_solar_generation) / wind_solar_generation_max

    return wind_solar_curtailment_rate

def take_out_leap_day(df):

  return df[~((df.index.month == 2) & (df.index.day == 29))]
