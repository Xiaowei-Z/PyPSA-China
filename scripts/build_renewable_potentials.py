import numpy as np
import pandas as pd
import os

from vresutils import reatlas as vreatlas, shapes as vshapes, landuse as vlanduse
from vresutils import cachable

cutout = vreatlas.Cutout(username='becker', cutoutname='Europe_2011_2014')
cutout.extent

ocean_depth_cutoff = 50 # in m or None

windturbines = dict(onshore='Vestas_V112_3MW', offshore='NREL_ReferenceTurbine_5MW_offshore')
solarpanel = dict(panel='KANENA', orientation='latitude_optimal')

partition = vreatlas.partition_from_emil(cutout).drop(['AL','ME','MK'])

onshoremap = cutout.meta['onshoremap']
offshoremap = (1-onshoremap)*(cutout.meta['heights'] > -ocean_depth_cutoff)

EUmask = np.asarray(list(partition)).any(axis=0)

EUonshoremap = onshoremap*EUmask
EUoffshoremap = offshoremap*EUmask

#The selection of CORINE Land Cover [1] types that are allowed for wind and solar are based on [2] p.42 / p.28
#
#[1] https://www.eea.europa.eu/ds_resolveuid/C9RK15EA06
#
#[2] Scholz, Y. (2012). Renewable energy based electricity supply at low costs: development of the REMix model and application for Europe.

lc_scholz_onshore = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32]
lc_scholz_solar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 31, 32]

lc_onshore = lc_scholz_onshore
lc_offshore = [44, 255]
lc_solar = lc_scholz_solar

#### $A_{RC}$: Raster cell area

reatlas_cell_areas=vlanduse._cutout_cell_areas(cutout)

#### $f_{LU}$: factor usable land area (via land use type and natura reserves)

@cachable
def get_landuse(cutout, lc, natura=True):
    return vlanduse.corine_for_cutout(cutout, lc, natura=natura)

onshore_landuse = get_landuse(cutout, lc_onshore, natura=True)
offshore_landuse = get_landuse(cutout, lc_offshore, natura=True)
solar_landuse = get_landuse(cutout, lc_solar, natura=True)

#### $G_s^{max inst}$ in units of MW/km$^2$

# ScholzPhD [2]
# Tab. 4.3.1: Area-specific installable capacity for on/offshore wind = 10MW/km^2
G_maxinst_onwind = 10.
G_maxinst_offwind = 10.
G_maxinst_solar = 170.

#### $f_{aa}$ : share of actually available area

f_aa_onwind = 0.2
f_aa_offwind = 0.2
f_aa_solar = 0.01

#### $uf$: utilization factor per raster cell per technology

# from generation.resource
import pytz
def capacity_factor_layout(resource, cutout, times_real_area=False):
    cf = cutout.reatlas.convert_and_aggregate(resource, save_sum=True) / len(cutout.meta['dates'])
    if set(('onshore', 'offshore')).issubset(resource):
        cf /= np.where(cutout.meta['onshoremap'],
                       vreatlas.windturbine_rated_capacity_per_unit(resource['onshore']),
                       vreatlas.windturbine_rated_capacity_per_unit(resource['offshore']))
    elif set(('panel', 'orientation')).issubset(resource):
        cf /= vreatlas.solarpanel_rated_capacity_per_m2(resource['panel'])
    else:
        raise NotImplemented

    if times_real_area:
        cf *= np.asarray(list(map(vshapes.area, cutout.grid_cells()))).reshape(cutout.shape)

    return cf

def unit_capacity_timeseries(resource, partition, capacitylayout, cutout, index=None, return_weight=False):
    if isinstance(partition, pd.Series):
        layouts = capacitylayout * np.asarray(list(partition))
        index = partition.index
    else:
        layouts = capacitylayout * partition

    reatlas = cutout.reatlas
    timesindex = pd.DatetimeIndex(cutout.meta['dates'], tz=pytz.utc)

    if set(('panel', 'orientation')).issubset(resource):
        rated_capacity_per_unit = vreatlas.solarpanel_rated_capacity_per_m2(resource['panel'])
    else:
        assert not set(('onshore', 'offshore')).issubset(resource), \
            "Only onshore or offshore is supported separately"
        turbine = resource.get('onshore') or resource.get('offshore')
        rated_capacity_per_unit = vreatlas.windturbine_rated_capacity_per_unit(turbine)
        resource = dict(onshore=turbine, offshore=turbine)
    weight = layouts.sum(axis=(1,2))
    timeseries = (reatlas.convert_and_aggregate(resource, layouts)
                  * np.nan_to_num(1./weight) / rated_capacity_per_unit)


    df = pd.DataFrame(timeseries, index=timesindex, columns=index)
    if return_weight:
        return df, weight
    else:
        return df

#%%time
windcapacityfactor = capacity_factor_layout(windturbines, cutout)

#%%time
solarcapacityfactor = capacity_factor_layout(solarpanel, cutout)

f_util_onwind = windcapacityfactor*EUonshoremap
f_util_offwind = windcapacityfactor*EUoffshoremap
f_util_solar = solarcapacityfactor*EUonshoremap

#### Layout $Y_{RC} \propto A_{RC} \cdot f_{LU} \cdot uf$

onshorelayout = reatlas_cell_areas * onshore_landuse * f_util_onwind
offshorelayout = reatlas_cell_areas * offshore_landuse * f_util_offwind
solarlayout = reatlas_cell_areas * solar_landuse * f_util_solar

beta = 1

def get_layouts(layout,partition,beta=1):
    partition_layouts = layout * np.asarray(list(partition))
    renormed_partition_layouts = np.nan_to_num(partition_layouts/ partition_layouts.max(axis=(1,2),keepdims=True))**beta
    return renormed_partition_layouts

onshorelayout_country = get_layouts(onshorelayout,partition,beta=beta)
offshorelayout_country = get_layouts(offshorelayout,partition,beta=beta)
solarlayout_country = get_layouts(solarlayout,partition,beta=beta)

##### split the onshore wind layout to comparable areas if countries are too large

def get_onwindpartition(partition,onshorelayout,max_parts=4,standardcountry='GR'):
    landarea=pd.Series([(reatlas_cell_areas*par*EUonshoremap).sum()
                        for par in partition.values],index=partition.keys())

    def mymaximum(self,maximum):
        return self.where(self<maximum,maximum)
    def atleast1(self):
        return self.where(self>1,1)

    landbits=mymaximum(atleast1((landarea/landarea[standardcountry]
                                ).round()).astype(np.int64)
                       ,max_parts) #limit e.g. FR to 4 pieces

    onwindpartition = partition.copy()
    for country in partition[landbits>1].index:
        onwindpartition.drop(country,inplace=True)

        par = partition[country]*EUonshoremap

        owl_= onshorelayout[par != 0]
        ncells = len(owl_)
        minmaxs = np.floor(np.linspace(0,ncells,landbits[country]+1)).astype(np.int64)
        minmaxs[-1] = -1
        bin_edges = np.sort(owl_)[minmaxs]
        bin_edges[-1] += 1.


        owl = onshorelayout*par
        for nn in np.arange(int(landbits[country])):
            onwindpartition.loc['{}{}'.format(country,int(nn))] = (((owl>=bin_edges[nn]) & (owl<bin_edges[nn+1])) * par) != 0

    return onwindpartition, landbits

onwindpartition, landbits = get_onwindpartition(partition,onshorelayout)

onshorelayout_country_split = get_layouts(onshorelayout,onwindpartition,beta=beta)

#### p_nom_max

#The capacity layout can only be scaled up until the first raster cell reaches the maximum installation density. Therefore, there exists a constant `const` for every node `n` such that:
#
#$const_n \cdot layout_n \le G^{max,inst} f_{aa} A_{RC} f_{LU} \qquad \forall RC \in n$
#
#The maximum value of `const` is then reached once
#
#$const_n = \min_{RC \in n} \frac{G^{max,inst} f_{aa} A_{RC} f_{LU}}{layout_n} $
#
#The maximum installable capacity `p_nom_max` is therefore:
#
#$p\_nom\_max_n = \sum_{RC \in n} const_n \cdot layout_n = \sum_{RC \in n} layout_n  \min_{RC \in n} \frac{A_{RC} f_{LU}}{layout_n} G^{max,inst} f_{aa}$

def get_p_nom_max(layout_country,partition,cell_areas,landuse,G_maxinst,f_aa):
    '''Return p_nom_max per country in partition.index
    Input
    -----
    layout :
        Relative distribution of generators.
    partition :
        partition
    '''
    mlayout = np.ma.array(layout_country,mask=(layout_country==0))

    p_nom_max = (mlayout.sum(axis=(1,2)) * G_maxinst * f_aa *
                 (cell_areas * landuse / mlayout).min(axis=(1,2))
                ) 
    return pd.Series(p_nom_max.data, index=partition.index)

dict_onwind = {'type':'onwind', 'partition':partition, 'layout':onshorelayout_country, 'landuse':onshore_landuse, 'G_maxinst':G_maxinst_onwind, 'f_aa':f_aa_onwind}
dict_onwind_split = {'type':'onwind_split', 'partition':onwindpartition, 'layout':onshorelayout_country_split, 'landuse':onshore_landuse, 'G_maxinst':G_maxinst_onwind, 'f_aa':f_aa_onwind}
dict_offwind = {'type':'offwind', 'partition':partition, 'layout':offshorelayout_country, 'landuse':offshore_landuse, 'G_maxinst':G_maxinst_offwind, 'f_aa':f_aa_offwind}
dict_solar = {'type':'solar', 'partition':partition, 'layout':solarlayout_country, 'landuse':solar_landuse, 'G_maxinst':G_maxinst_solar, 'f_aa':f_aa_solar}


p_nom_max_folder = 'data/renewables/store_p_nom_max/'

if not os.path.isdir(p_nom_max_folder):
    os.makedirs(p_nom_max_folder)

for typ in [dict_onwind, dict_onwind_split, dict_offwind, dict_solar]:
    p_nom_max = get_p_nom_max(typ['layout'],typ['partition'],reatlas_cell_areas,typ['landuse'],typ['G_maxinst'],typ['f_aa'])
    
    p_nom_max_file = os.path.join(p_nom_max_folder,'p_nom_max_{typ}_beta{beta}.pickle'.format(typ=typ['type'],beta=beta))

    print('saving file {}'.format(p_nom_max_file))
    p_nom_max.to_pickle(p_nom_max_file)

#### p_max_pu time series

#%%time
onshore = unit_capacity_timeseries(dict(onshore=windturbines['onshore']), partition,
                                            onshorelayout_country, cutout)

#%%time
onshore_split = unit_capacity_timeseries(dict(onshore=windturbines['onshore']), onwindpartition,
                                         onshorelayout_country_split, cutout)

#%%time
offshore = unit_capacity_timeseries(dict(offshore=windturbines['offshore']), partition,
                                    offshorelayout_country, cutout)

#%%time
solar = unit_capacity_timeseries(solarpanel, partition, solarlayout_country, cutout)

p_max_pus = dict(onwind=onshore,onwind_split=onshore_split,offwind=offshore,solar=solar)


p_max_pu_folder='data/renewables/store_p_max_pu_betas/'

if not os.path.isdir(p_max_pu_folder):
    os.makedirs(p_max_pu_folder)

for kind, pmpu in p_max_pus.iteritems():
    pmpu_file = os.path.join(p_max_pu_folder,'p_max_pu_{kind}_beta{beta}.pickle'.format(kind=kind,beta=beta))
    print('saving file: {}'.format(pmpu_file))
    pmpu.to_pickle(pmpu_file)

splitcountries=landbits[landbits>1] # name of splitted countries and number of parts
splitcountries_filename = os.path.join(p_max_pu_folder,'onwind_split_countries.csv')
print('saving country splits to: {}'.format(splitcountries_filename))
splitcountries.to_csv(splitcountries_filename)


