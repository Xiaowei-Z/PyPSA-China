# from pyutilib.services import TempfileManager
# TempfileManager.tempdir = '/home/201402677/tmp'

from vresutils.costdata import annuity
import vresutils.hydro as vhydro
import vresutils.file_io_helper as io_helper
import vresutils.load as vload
import vresutils.shapes as vshapes
from vresutils import timer

import pypsa
from shapely.geometry import Point
import geopandas as gpd
import datetime
import pandas as pd
import numpy as np
import os
import pytz
import yaml
from six import iteritems, iterkeys, itervalues
import sys
from math import radians, cos, sin, asin, sqrt
from functools import partial
import pyproj
from shapely.ops import transform
import warnings
import helper

from pyomo.environ import Constraint

from functions import pro_names


#This function follows http://toblerity.org/shapely/manual.html
def area_from_lon_lat_poly(geometry):
    """For shapely geometry in lon-lat coordinates,
    returns area in km^2."""

    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), # Source: Lon-Lat
        pyproj.Proj(proj='aea')) # Target: Albers Equal Area Conical https://en.wikipedia.org/wiki/Albers_projection

    new_geometry = transform(project, geometry)

    #default area is in m^2
    return new_geometry.area/1e6

def get_p_max_pu(path, timerange):

    pmpu = pd.DataFrame(np.load(path), index=pd.date_range('1979-01-01 00:00:00', '2016-12-31 23:00:00', freq='H'), columns=pro_names)
    pmpu = pmpu.loc[timerange, :]

    return pmpu

def haversine(p1,p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def generate_periodic_profiles(dt_index=None,col_tzs=pd.Series(index=pro_names, data=len(pro_names)*['Shanghai']),weekly_profile=range(24*7)):
    """Give a 24*7 long list of weekly hourly profiles, generate this
    for each country for the period dt_index, taking account of time
    zones and Summer Time."""


    weekly_profile = pd.Series(weekly_profile,range(24*7))

    week_df = pd.DataFrame(index=dt_index,columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [24*dt.weekday()+dt.hour for dt in dt_index.tz_convert(pytz.timezone("Asia/{}".format(col_tzs[ct])))]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df

def shift_df(df,hours=1):
    """Works both on Series and DataFrame"""
    df = df.copy()
    df.values[:] = np.concatenate([df.values[-hours:],
                                   df.values[:-hours]])
    return df

def transport_degree_factor(temperature,deadband_lower=15,deadband_upper=20,
    lower_degree_factor=0.5,
    upper_degree_factor=1.6):
    """Work out how much energy demand in vehicles increases due to heating and cooling.
    There is a deadband where there is no increase.
    Degree factors are % increase in demand compared to no heating/cooling fuel consumption.
    Returns per unit increase in demand for each place and time"""

    dd = temperature.copy()

    dd[(temperature > deadband_lower) & (temperature < deadband_upper)] = 0.

    dd[temperature < deadband_lower] = lower_degree_factor/100.*(deadband_lower-temperature[temperature < deadband_lower])

    dd[temperature > deadband_upper] = upper_degree_factor/100.*(temperature[temperature > deadband_upper]-deadband_upper)

    return dd

def prepare_data(network):


    ##############
    #Heating
    ##############

    #copy forward the daily average heat demand into each hour, so it can be multipled by the intraday profile

    with pd.HDFStore(snakemake.input.heat_demand_name, mode='r') as store:
        #the ffill converts daily values into hourly values
        h = store['heat_demand_profiles']
        h_n = h[~h.index.duplicated(keep='first')].iloc[:-1,:]
        heat_demand_hdh = h_n.reindex(index=network.snapshots, method="ffill")

    with pd.HDFStore(snakemake.input.cop_name, mode='r') as store:
        ashp_cop = store['ashp_cop_profiles'].reindex(index=network.snapshots)
        gshp_cop = store['gshp_cop_profiles'].reindex(index=network.snapshots)

    with pd.HDFStore(snakemake.input.energy_totals_name, mode='r') as store:
        space_heating_per_hdd = store['space_heating_per_hdd']
        hot_water_per_day = store['hot_water_per_day']

    intraday_profiles = pd.read_csv("data/heating/heat_load_profile_DK_AdamJensen.csv",index_col=0)
    intraday_year_profiles = generate_periodic_profiles(dt_index=heat_demand_hdh.index.tz_localize("UTC"), weekly_profile=(list(intraday_profiles["weekday"])*5 + list(intraday_profiles["weekend"])*2)).tz_localize(None)

    space_heat_demand = intraday_year_profiles.mul(heat_demand_hdh).mul(space_heating_per_hdd)
    water_heat_demand = intraday_year_profiles.mul(hot_water_per_day/24.)

    heat_demand = space_heat_demand + water_heat_demand

    ###############
    #CO2
    ###############

    #tCO2
    represented_hours = network.snapshot_weightings.sum()
    Nyears= represented_hours/8760.
    with pd.HDFStore(snakemake.input.co2_totals_name, mode='r') as store:
        co2_totals = Nyears * store['co2']

    ###############
    #renewables
    ###############

    #load renewables time series
    variable_generator_kinds = {'onwind':'onwind','offwind':'offwind','solar':'solar'}
    if options['split_onwind']:
        variable_generator_kinds.update({'onwind':'onwind_split'})

    p_max_pu_folder = 'data/p_max_pu/'
    #dict of dfs with index datetime and col node names
    p_max_pu = {kind: get_p_max_pu(p_max_pu_folder + kname + '.npy', network.snapshots) for kind, kname in variable_generator_kinds.items()}

    p_nom_max_folder = 'data/renewable_potential/'
    #dict of series with index node names
    p_nom_max = {kind: pd.read_pickle(p_nom_max_folder + kname + '.pickle') for kind, kname in variable_generator_kinds.items()}


    return heat_demand, space_heat_demand, water_heat_demand, ashp_cop, gshp_cop, co2_totals, p_max_pu, p_nom_max


def prepare_network(options):

    #Build the Network object, which stores all other objects
    network = helper.Network()

    network.options=options

    #load graph
    nodes = pd.Index(pro_names)
    edges = pd.read_csv("data/graph/edges.txt", sep=",", header=None)

    #set times
    network.set_snapshots(pd.date_range(options['tmin'],options['tmax'],freq='H'))

    represented_hours = network.snapshot_weightings.sum()
    Nyears= represented_hours/8760.

    #set all asset costs and other parameters
    costs = pd.read_csv("data/costs/costs.csv",index_col=list(range(3))).sort_index()

    #correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"]*=1e3
    costs.loc[costs.unit.str.contains("USD"),"value"]*=options['USD2013_to_EUR2013']

    cost_year = 2030

    costs = costs.loc[pd.IndexSlice[:,cost_year,:],"value"].unstack(level=2).groupby(level="technology").sum(min_count=1)

    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : options['discountrate'],
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : 25
    })

    costs["fixed"] = [(annuity(v["lifetime"],v["discount rate"])+v["FOM"]/100.)*v["investment"]*Nyears for i,v in costs.iterrows()]

    heat_demand, space_heat_demand, water_heat_demand, ashp_cop, gshp_cop, co2_totals, p_max_pu, p_nom_max = prepare_data(network)


    pro_shapes = gpd.GeoDataFrame.from_file('data/province_shapes/CHN_adm1.shp')
    pro_shapes.index = pro_names

    # add buses
    network.madd('Bus',
        nodes,
        x=pro_shapes['geometry'].centroid.x,
        y=pro_shapes['geometry'].centroid.y,
        )

    #add carriers
    network.add("Carrier","gas",co2_emissions=costs.at['gas','CO2 intensity']) # in t_CO2/MWht
    network.add("Carrier","onwind")
    network.add("Carrier","offwind")
    network.add("Carrier","solar")
    if options['add_PHS']:
        network.add("Carrier","PHS")
    if options['add_hydro']:
        network.add("Carrier","hydro")
    if options['add_ror']:
        network.add("Carrier","ror")
    if options['add_H2_storage']:
        network.add("Carrier","H2")
    if options['add_battery_storage']:
        network.add("Carrier","battery")
    if options["heat_coupling"]:
        network.add("Carrier","heat")
        network.add("Carrier","water tanks")
    if options["retrofitting"]:
        network.add("Carrier", "retrofitting")
    if options["transport_coupling"]:
        network.add("Carrier","Li ion")


    if options['co2_reduction'] is not None:

        co2_limit = co2_totals["electricity"]

        if options["transport_coupling"]:
            co2_limit += co2_totals['transport']

        if options["heat_coupling"]:
            co2_limit += co2_totals['heating']

        co2_limit *= Nyears * (1 - options['co2_reduction'])

        network.add("GlobalConstraint",
                    "co2_limit",
                    type="primary_energy",
                    carrier_attribute="co2_emissions",
                    sense="<=",
                    constant=co2_limit)


    #load demand data
    with pd.HDFStore('data/load/load_2016_weatheryears_1979_2016_TWh_UTCtime.h5', mode='r') as store:
        load = 1e6 * store['load'].loc[network.snapshots]

    load.columns = pro_names

    network.madd("Load", nodes, bus=nodes, p_set=load[nodes])

    #add renewables
    network.madd("Generator",
                 nodes,
                 suffix=' onwind',
                 bus=nodes,
                 p_nom_extendable=True,
                 carrier="onwind",
                 p_nom_max=p_nom_max['onwind'][nodes],
                 capital_cost = costs.at['onwind','fixed'],
                 marginal_cost=costs.at['onwind','VOM'],
                 p_max_pu=p_max_pu['onwind'][nodes])

    offwind_nodes = p_nom_max['offwind'][p_nom_max['offwind']!=0].index
    network.madd("Generator",
                 offwind_nodes,
                 suffix=' offwind',
                 p_nom_extendable=True,
                 bus=offwind_nodes,
                 carrier="offwind",
                 p_nom_max=p_nom_max['offwind'][offwind_nodes],
                 capital_cost = costs.at['offwind','fixed'],
                 p_max_pu=p_max_pu['offwind'][offwind_nodes],
                 marginal_cost=costs.at['offwind','VOM'])

    network.madd("Generator",
                 nodes,
                 suffix=' solar',
                 p_nom_extendable=True,
                 bus=nodes,
                 carrier="solar",
                 p_nom_max=p_nom_max['solar'][nodes],
                 capital_cost = 0.5*(costs.at['solar-rooftop','fixed']+costs.at['solar-utility','fixed']),
                 p_max_pu=p_max_pu['solar'][nodes].clip(1.e-5),
                 marginal_cost=costs.at['solar','VOM'])

    #add conventionals
    for generator,carrier in [("OCGT","gas")]:
        # add converter from fuel source

        network.add("Bus",
                    "EU " + carrier,
                    carrier=carrier)

        network.madd("Link",
                     nodes + " " + generator,
                     bus0=["EU " + carrier]*len(nodes),
                     bus1=nodes,
                     marginal_cost=costs.at[generator,'efficiency']*costs.at[generator,'VOM'], #NB: VOM is per MWel
                     capital_cost=costs.at[generator,'efficiency']*costs.at[generator,'fixed'], #NB: fixed cost is per MWel
                     p_nom_extendable=True,
                     efficiency=costs.at[generator,'efficiency'])

        network.add("Store",
                    "EU " + carrier + " Store",
                    bus="EU " + carrier,
                    e_nom_extendable=True,
                    e_min_pu=-1.,
                    marginal_cost=costs.at[carrier,'fuel'])

    if options['nuclear']:
        network.add("Carrier","uranium")

        network.madd("Generator",
                     nodes,
                     suffix=' nuclear',
                     p_nom_extendable=True,
                     bus=nodes,
                     carrier="uranium",
                     efficiency=costs.at['nuclear','efficiency'],
                     capital_cost = costs.at['nuclear','fixed'],
                     marginal_cost=costs.at['nuclear','VOM'] + costs.at['uranium','fuel']/costs.at['nuclear','efficiency'])

    if options['add_PHS']:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        phss = hydrocapa_df.index[hydrocapa_df['p_nom_store[GW]'] > 0].intersection(nodes)
        if options['hydro_capital_cost']:
            cc=costs.at['PHS','fixed']
        else:
            cc=0.

        network.madd("StorageUnit",
                     phss,
                     suffix=" PHS",
                     bus=phss,
                     carrier="PHS",
                     p_nom_extendable=False,
                     p_nom=hydrocapa_df.loc[phss]['p_nom_store[GW]']*1000., #from GW to MW
                     max_hours=options['PHS_max_hours'],
                     efficiency_store=np.sqrt(costs.at['PHS','efficiency']),
                     efficiency_dispatch=np.sqrt(costs.at['PHS','efficiency']),
                     cyclic_state_of_charge=True,
                     capital_cost = cc,
                     marginal_cost=options['marginal_cost_storage'])

    if options['add_hydro']:

        #######
        df = pd.read_csv('data/hydro/dams_large.csv', index_col=0)
        points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
        dams = gpd.GeoDataFrame(df, geometry=points)
        dams.crs = {'init': 'epsg:4326'}

        hourly_rng = pd.date_range('1979-01-01', '2017-01-01', freq='1H', closed='left')
        inflow = pd.read_pickle('data/hydro/daily_hydro_inflow_per_dam_1979_2016_m3.pickle').reindex(hourly_rng, fill_value=0)
        inflow.columns = dams.index

        water_consumption_factor = dams.loc[:, 'Water_consumption_factor_avg'] * 1e3 # m^3/KWh -> m^3/MWh

        #######
        # ### Add hydro stations as buses
        network.madd('Bus',
            dams.index,
            suffix=' station',
            carrier='stations',
            x=dams['geometry'].centroid.x,
            y=dams['geometry'].centroid.y);

        dam_buses = network.buses[network.buses.carrier=='stations']


        # ### add hydro reservoirs as stores

        initial_capacity = pd.read_pickle('data/hydro/reservoir_initial_capacity.pickle')
        total_capacity = pd.read_pickle('data/hydro/reservoir_total_capacity.pickle')
        effective_capacity = pd.read_pickle('data/hydro/reservoir_effective_capacity.pickle')
        initial_capacity.index = dams.index
        total_capacity.index = dams.index
        effective_capacity.index = dams.index

        network.madd('Store',
            dams.index,
            suffix=' reservoir',
            bus=dam_buses.index,
            e_nom=effective_capacity,
            e_initial=(effective_capacity - (total_capacity - initial_capacity)),
            e_cyclic=True,
            marginal_cost=options['marginal_cost_storage']);


        # ### add hydro turbines to link stations to provinces
        network.madd('Link',
                    dams.index,
                    suffix=' turbines',
                    bus0=dam_buses.index,
                    bus1=dams['Province'],
                    p_nom=10 * dams['installed_capacity_10MW'] / (1 / water_consumption_factor),
                    efficiency= 1 / water_consumption_factor);
        # p_nom * efficiency = 10 * dams['installed_capacity_10MW']


        # ### add rivers to link station to station
        bus0s = [0, 21, 11, 19, 22, 29, 8, 40, 25, 1, 7, 4, 10, 15, 12, 20, 26, 6, 3, 39]
        bus1s = [5, 11, 19, 22, 32, 8, 40, 25, 35, 2, 4, 10, 9, 12, 20, 23, 6, 17, 14, 16]

        for bus0, bus2 in list(zip(dams.index[bus0s], dam_buses.iloc[bus1s].index)):

            # normal flow
            network.links.at[bus0 + ' turbines', 'bus2'] = bus2
            network.links.at[bus0 + ' turbines', 'efficiency2'] = 1.

        #### spillage
        for bus0, bus1 in list(zip(dam_buses.iloc[bus0s].index, dam_buses.iloc[bus1s].index)):
            network.add('Link',
                       "{}-{}".format(bus0,bus1) + ' spillage',
                       bus0=bus0,
                       bus1=bus1,
                       p_nom_extendable=True)

        dam_ends = [dam for dam in range(len(dams.index)) if (dam in bus1s and dam not in bus0s) or (dam not in bus0s+bus1s)]

        for bus0 in dam_buses.iloc[dam_ends].index:
            network.add('Link',
                        bus0 + ' spillage',
                        bus0=bus0,
                        bus1='Tibet',
                        p_nom_extendable=True,
                        efficiency=0.0)

        #### add inflow as generators
        # only feed into hydro stations which are the first of a cascade
        inflow_stations = [dam for dam in range(len(dams.index)) if not dam in bus1s ]

        for inflow_station in inflow_stations:

            # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow
            p_nom = inflow.loc[network.snapshots].iloc[:,inflow_station].max()
            p_pu = inflow.loc[network.snapshots].iloc[:,inflow_station] / p_nom

            network.add('Generator',
                       dams.index[inflow_station] + ' inflow',
                       bus=dam_buses.iloc[inflow_station].name,
                       carrier='hydro_inflow',
                       p_max_pu=p_pu,
                       p_min_pu=p_pu,
                       p_nom=p_nom)
            # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power


        # ### add fake hydro just to introduce capital cost
        if options['add_hydro'] and options['hydro_capital_cost']:
            hydro_cc=costs.at['hydro','fixed']
        else: hydro_cc=0.

        network.madd('StorageUnit',
            dams.index,
            suffix=' hydro dummy',
            bus=dams['Province'],
            carrier='hydro',
            p_nom=10 * dams['installed_capacity_10MW'],
            p_max_pu=0.,
            p_min_pu=0.,
            capital_cost=hydro_cc)

    if options['add_ror']:
        rors = ror_share.index[ror_share > 0.]
        rors = rors.intersection(nodes)
        rors = rors.intersection(inflow_df.columns)
        pnom = ror_share[rors]*hydrocapa_df.loc[rors,'p_nom_discharge[GW]']*1000. #GW to MW
        inflow_pu = inflow_df[rors].multiply(ror_share[rors]/pnom)
        inflow_pu[inflow_pu>1]=1. #limit inflow per unit to one, i.e, excess inflow is spilled here

        if options['hydro_capital_cost']:
            cc=costs.at['ror','fixed']
        else:
            cc=0.

        network.madd("Generator",
                     rors,
                     suffix=" ror",
                     bus=rors,
                     carrier="ror",
                     p_nom_extendable=False,
                     p_nom=pnom,
                     p_max_pu=inflow_pu,
                     capital_cost = cc,
                     marginal_cost=options['marginal_cost_storage'])

    if options['add_H2_storage']:

        network.madd("Bus",
                     nodes+ " H2",
                     carrier="H2")

        network.madd("Link",
                    nodes + " H2 Electrolysis",
                    bus1=nodes + " H2",
                    bus0=nodes,
                    p_nom_extendable=True,
                    efficiency=costs.at["electrolysis","efficiency"],
                    capital_cost=costs.at["electrolysis","fixed"])

        network.madd("Link",
                     nodes + " H2 Fuel Cell",
                     bus0=nodes + " H2",
                     bus1=nodes,
                     p_nom_extendable=True,
                     efficiency=costs.at["fuel cell","efficiency"],
                     capital_cost=costs.at["fuel cell","fixed"]*costs.at["fuel cell","efficiency"])  #NB: fixed cost is per MWel

        network.madd("Store",
                     nodes + " H2 Store",
                     bus=nodes + " H2",
                     e_nom_extendable=True,
                     e_cyclic=True,
                     capital_cost=costs.at["hydrogen storage","fixed"])

    if options['add_methanation']:
        network.madd("Link",
                     nodes + " Sabatier",
                     bus0=nodes+" H2",
                     bus1=["EU " + carrier]*len(nodes),
                     p_nom_extendable=True,
                     efficiency=costs.at["methanation","efficiency"],
                     capital_cost=costs.at["methanation","fixed"])

    if options['helmeth']:
        network.madd("Link",
                     nodes + " helmeth",
                     bus0=nodes,
                     bus1=["EU " + carrier]*len(nodes),
                     p_nom_extendable=True,
                     efficiency=costs.at["helmeth","efficiency"],
                     capital_cost=costs.at["helmeth","fixed"])

    if options['add_battery_storage']:

        network.madd("Bus",
                     nodes + " battery",
                     carrier="battery")

        network.madd("Store",
                     nodes + " battery",
                     bus=nodes + " battery",
                     e_cyclic=True,
                     e_nom_extendable=True,
                     capital_cost=costs.at['battery storage','fixed'])

        network.madd("Link",
                     nodes + " battery charger",
                     bus0=nodes,
                     bus1=nodes + " battery",
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     capital_cost=costs.at['battery inverter','fixed'],
                     p_nom_extendable=True)

        network.madd("Link",
                     nodes + " battery discharger",
                     bus0=nodes + " battery",
                     bus1=nodes,
                     efficiency=costs.at['battery inverter','efficiency']**0.5,
                     marginal_cost=options['marginal_cost_storage'],
                     p_nom_extendable=True)

    #Sources:
    #[HP]: Henning, Palzer http://www.sciencedirect.com/science/article/pii/S1364032113006710
    #[B]: Budischak et al. http://www.sciencedirect.com/science/article/pii/S0378775312014759

    if options["heat_coupling"]:

        # urban = nodes

        #NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

        #central are urban nodes with district heating
        # central = nodes ^ urban

        central_fraction = pd.read_hdf("data/heating/DH_percent.h5")

        for cat in [' decentral ', ' central ']:

            network.madd("Bus",
                         nodes + cat + "heat",
                         carrier="heat")

            network.madd('Generator',
                          nodes+cat+'dummy heaters',
                          bus=nodes+cat+'heat',
                          p_nom_extendable=True,
                          )

        network.madd("Load",
                     nodes,
                     suffix=" decentral heat",
                     bus=nodes + " decentral heat",
                     p_set= heat_demand[nodes].multiply((1-central_fraction)))

        network.madd("Load",
                     nodes,
                     suffix=" central heat",
                     bus=nodes + " central heat",
                     p_set= heat_demand[nodes].multiply(central_fraction))


        if options['heat_pumps']:

            for cat in [' decentral ', ' central ']:
                network.madd("Link",
                             nodes,
                             suffix=cat + "heat pump",
                             bus0=nodes,
                             bus1=nodes + cat + "heat",
                             efficiency=ashp_cop[nodes] if options["time_dep_hp_cop"] else costs.at[cat.lstrip()+"air-sourced heat pump",'efficiency'],
                             capital_cost=costs.at[cat.lstrip()+'air-sourced heat pump','efficiency']*costs.at[cat.lstrip()+'air-sourced heat pump','fixed'],
                             p_nom_extendable=True)

            network.madd("Link",
                         nodes,
                         suffix=" ground heat pump",
                         bus0=nodes,
                         bus1=nodes + " decentral heat",
                         efficiency=gshp_cop[nodes] if options["time_dep_hp_cop"] else costs.at['decentral ground-sourced heat pump','efficiency'],
                         capital_cost=costs.at['decentral ground-sourced heat pump','efficiency']*costs.at['decentral ground-sourced heat pump','fixed'],
                         p_nom_extendable=True)

        if options['retrofitting']:

            retro_nodes = pd.Index(["DE"])

            space_heat_demand = space_heat_demand[retro_nodes]

            square_metres = population[retro_nodes]/population['DE']*5.7e9   #HPI 3.4e9m^2 for DE res, 2.3e9m^2 for tert https://doi.org/10.1016/j.rser.2013.09.012

            space_peak = space_heat_demand.max()

            space_pu = space_heat_demand.divide(space_peak)

            network.madd('Generator',
                         retro_nodes,
                         suffix=' retrofitting I',
                         bus=retro_nodes+' heat',
                         carrier="retrofitting",
                         p_nom_extendable=True,
                         p_nom_max=options['retroI-fraction']*space_peak*(1-central_fraction),
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting I','fixed']*square_metres/(options['retroI-fraction']*space_peak))

            network.madd('Generator',
                         retro_nodes,
                         suffix=' retrofitting II',
                         bus=retro_nodes+' heat',
                         carrier="retrofitting",
                         p_nom_extendable=True,
                         p_nom_max=options['retroII-fraction']*space_peak*(1-central_fraction),
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting II','fixed']*square_metres/(options['retroII-fraction']*space_peak))

            network.madd('Generator',
                         retro_nodes,
                         suffix=' urban retrofitting I',
                         bus=retro_nodes+' urban heat',
                         carrier="retrofitting",
                         p_nom_extendable=True,
                         p_nom_max=options['retroI-fraction']*space_peak*central_fraction,
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting I','fixed']*square_metres/(options['retroI-fraction']*space_peak))

            network.madd('Generator',
                         retro_nodes,
                         suffix=' urban retrofitting II',
                         bus=retro_nodes+' urban heat',
                         carrier="retrofitting",
                         p_nom_extendable=True,
                         p_nom_max=options['retroII-fraction']*space_peak*central_fraction,
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting II','fixed']*square_metres/(options['retroII-fraction']*space_peak))

        if options["tes"]:

            for cat in [' decentral ', ' central ']:
                network.madd("Bus",
                            nodes + cat + "water tanks",
                            carrier="water tanks")

                network.madd("Link",
                             nodes + cat + "water tanks charger",
                             bus0=nodes + cat + "heat",
                             bus1=nodes + cat + "water tanks",
                             efficiency=costs.at['water tank charger','efficiency'],
                             p_nom_extendable=True)

                network.madd("Link",
                             nodes + cat + "water tanks discharger",
                             bus0=nodes + cat + "water tanks",
                             bus1=nodes + cat + "heat",
                             efficiency=costs.at['water tank discharger','efficiency'],
                             p_nom_extendable=True)

                network.madd("Store",
                             nodes + cat + "water tank",
                             bus=nodes + cat + "water tanks",
                             e_cyclic=True,
                             e_nom_extendable=True,
                             standing_loss=1-np.exp(-1/(24.*options["tes_tau"])),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                             capital_cost=costs.at[cat.lstrip()+'water tank storage','fixed']/(1.17e-3*40)) #conversion from EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K

        if options["boilers"]:

            for cat in [' decentral ', ' central ']:
                network.madd("Link",
                             nodes + cat + "resistive heater",
                             bus0=nodes,
                             bus1=nodes + cat + "heat",
                             efficiency=costs.at[cat.lstrip()+'resistive heater','efficiency'],
                             capital_cost=costs.at[cat.lstrip()+'resistive heater','efficiency']*costs.at[cat.lstrip()+'resistive heater','fixed'],
                             p_nom_extendable=True)

                network.madd("Link",
                             nodes + cat + "gas boiler",
                             p_nom_extendable=True,
                             bus0=["EU " + carrier]*len(nodes),
                             bus1=nodes + cat + "heat",
                             efficiency=costs.at[cat.lstrip()+'gas boiler','efficiency'],
                             capital_cost=costs.at[cat.lstrip()+'gas boiler','efficiency']*costs.at[cat.lstrip()+'gas boiler','fixed'])

        if options["chp"]:

            network.madd("CHP",
                         nodes + " central CHP",
                         bus_source=["EU " + carrier]*len(nodes),
                         bus_elec=nodes,
                         bus_heat=nodes + " central heat",
                         p_nom_extendable=True,
                         capital_cost=costs.at['central CHP','fixed'],
                         eta_elec=options['chp_parameters']['eta_elec'],
                         c_v=options['chp_parameters']['c_v'],
                         c_m=options['chp_parameters']['c_m'],
                         p_nom_ratio=options['chp_parameters']['p_nom_ratio'])

        if options["solar_thermal"]:

            #this is the amount of heat collected in W per m^2, accounting
            #for efficiency
            with pd.HDFStore(snakemake.input.solar_thermal_name, mode='r') as store:
                #1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
                solar_thermal = options['solar_cf_correction'] * store['solar_thermal_profiles']/1e3

            for cat in [' decentral ', ' central ']:
                network.madd("Generator",
                             nodes,
                             suffix=cat + "solar thermal collector",
                             bus=nodes + cat + "heat",
                             carrier="solar",
                             p_nom_extendable=True,
                             capital_cost=costs.at[cat.lstrip()+'solar thermal','fixed'],
                             p_max_pu=solar_thermal[nodes])


    if options['dac']: # Direct Air Capture

        network.add("Carrier",
                    "co2",
                    co2_emissions=1.)

        network.add("Bus",
                    "EU co2",
                    carrier="co2")

        #could add capital costs here
        network.add("Store",
                    "EU co2 Store",
                    bus="EU co2",
                    e_nom_extendable=True)

        #could consider to do this in high density area
        network.madd("Link",
                     nodes + " DAC",
                     bus0=["EU co2"]*len(nodes),
                     bus1=nodes + " decentral heat",
                     bus2=nodes,
                     p_max_pu=0,
                     p_min_pu=-1,
                     p_nom_extendable=True,
                     efficiency=1.5,
                     efficiency2=0.22,
                     capital_cost=costs.at["DAC","fixed"]*8760)

    if options["transport_coupling"]:

        network.madd("Bus",
                     nodes,
                     suffix=" EV battery",
                     carrier="Li ion")

        network.madd("Load",
                     nodes,
                     suffix=" transport",
                     bus=nodes + " EV battery",
                     p_set=(1-options['transport_fuel_cell_share'])*(transport[nodes]+shift_df(transport[nodes],1)+shift_df(transport[nodes],2))/3.)

        p_nom = transport_data["number cars"]*0.011*(1-options['transport_fuel_cell_share'])  #3-phase charger with 11 kW * x% of time grid-connected

        network.madd("Link",
                     nodes,
                     suffix= " BEV charger",
                     bus0=nodes,
                     bus1=nodes + " EV battery",
                     p_nom=p_nom,
                     p_max_pu=avail_profile[nodes],
                     efficiency=0.9, #[B]
                     #These were set non-zero to find LU infeasibility when availability = 0.25
                     #p_nom_extendable=True,
                     #p_nom_min=p_nom,
                     #capital_cost=1e6,  #i.e. so high it only gets built where necessary
                     )

        if options["v2g"]:

            network.madd("Link",
                         nodes,
                         suffix=" V2G",
                         bus1=nodes,
                         bus0=nodes + " EV battery",
                         p_nom=p_nom,
                         p_max_pu=avail_profile[nodes],
                         efficiency=0.9)  #[B]



        if options["bev"]:

            network.madd("Store",
                         nodes,
                         suffix=" battery storage",
                         bus=nodes + " EV battery",
                         e_cyclic=True,
                         e_nom=transport_data["number cars"]*0.05*options["bev_availability"]*(1-options['transport_fuel_cell_share']), #50 kWh battery http://www.zeit.de/mobilitaet/2014-10/auto-fahrzeug-bestand
                         e_max_pu=1,
                         e_min_pu=dsm_profile[nodes])


        if options['transport_fuel_cell_share'] != 0:

            network.madd("Load",
                         nodes,
                         suffix=" transport fuel cell",
                         bus=nodes + " H2",
                         p_set=options['transport_fuel_cell_share']/0.58*transport[nodes])

    #add lines
    if not network.options['no_lines']:

        lengths = np.array([haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                      [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]) for name0,name1 in edges.values])

        if options['line_volume_limit_factor'] is not None:
            cc = Nyears*0.01 # Set line costs to ~zero because we already restrict the line volume
        else:
            cc = ((options['line_cost_factor']*lengths*costs.at['HVDC overhead','fixed']*1.25+costs.at['HVDC inverter pair','fixed']) \
                    * 1.5)
            # 1.25 because lines are not straight, 150000 is per MW cost of
            # converter pair for DC line,
            # n-1 security is approximated by an overcapacity factor 1.5 ~ 1./0.666667
            #FOM of 2%/a


        network.madd("Link",
                     edges[0] + '-' + edges[1],
                     bus0=edges[0].values,
                     bus1=edges[1].values,
                     p_nom_extendable=True,
                     p_min_pu=-1,
                     length=lengths,
                     capital_cost=cc)

    return network

if __name__ == '__main__':

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        with open('config.yaml') as f:
            config = yaml.load(f)
        snakemake = Dict()
        snakemake.input = Dict(options_name=config['results_dir'] + 'version-' + str(config['version']) + '/options/options-{flexibility}-{line_limits}-{co2_reduction}.yml',
            population_name='data/population.h5',
            solar_thermal_name='data/heating/solar_thermal-{angle}.h5'.format(angle=config['solar_thermal_angle']),
            heat_demand_name='data/heating/daily_heat_demand.h5',
            cop_name='data/heating/cop.h5',
            energy_totals_name='data/energy_totals.h5',
            co2_totals_name='data/co2_totals.h5',
            temp='data/heating/temp.h5',)
        snakemake.output = Dict(network_name=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{flexibility}-{line_limits}-{co2_reduction}.nc')

    options = yaml.load(open(snakemake.input.options_name,"r"))

    population = pd.read_hdf(snakemake.input.population_name)

    network = prepare_network(options)

    network.export_to_netcdf(snakemake.output.network_name)
