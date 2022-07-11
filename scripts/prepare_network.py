# from pyutilib.services import TempfileManager
# TempfileManager.tempdir = '/home/201402677/tmp'

from vresutils.costdata import annuity
# import vresutils.hydro as vhydro
# import vresutils.file_io_helper as io_helper
# import vresutils.load as vload
# import vresutils.shapes as vshapes
# from vresutils import timer

import logging
from _helpers import configure_logging

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
import xarray as xr

from functions import pro_names, HVAC_cost_curve


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


def prepare_costs(Nyears, config):

    #set all asset costs and other parameters
    costs = pd.read_csv("data/costs.csv",index_col=list(range(3))).sort_index()

    #correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"]*=1e3
    costs.loc[costs.unit.str.contains("USD"),"value"]*=config['costs']['USD2013_to_EUR2013']
    # scale co2 intensity by 10^6 to reduce matrix range
    costs.loc[costs.unit.str.contains("tCO2/MWhth"),"value"]*=1.e-3
    costs.loc[costs.unit.str.contains("tCO2/MWhth"),"unit"] = 'kilo tCO2/MWhth'

    cost_year = 2030

    costs = costs.loc[pd.IndexSlice[:,cost_year,:],"value"].unstack(level=2).groupby(level="technology").sum(min_count=1)

    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : config['costs']['discountrate'],
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : 25
    })

    costs["fixed"] = [(annuity(v["lifetime"],v["discount rate"])+v["FOM"]/100.)*v["investment"]*Nyears for i,v in costs.iterrows()]

    return costs


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

    heat_demand = space_heat_demand + water_heat_demand#only consider heat demand at first

    ###############
    #CO2
    ###############

    #tCO2
    represented_hours = network.snapshot_weightings.sum()[0]
    Nyears= represented_hours/8760.
    with pd.HDFStore(snakemake.input.co2_totals_name, mode='r') as store:
        co2_totals = Nyears * store['co2']

    return heat_demand, space_heat_demand, water_heat_demand, ashp_cop, gshp_cop, co2_totals


def prepare_network(config):
    # add CHP definition
    override_component_attrs = pypsa.descriptors.Dict(
        {k: v.copy() for k, v in pypsa.components.component_attrs.items()}
    )
    override_component_attrs["Link"].loc["bus2"] = [
        "string",
        np.nan,
        np.nan,
        "2nd bus",
        "Input (optional)",
    ]
    override_component_attrs["Link"].loc["efficiency2"] = [
        "static or series",
        "per unit",
        1.0,
        "2nd bus efficiency",
        "Input (optional)",
    ]
    override_component_attrs["Link"].loc["p2"] = [
        "series",
        "MW",
        0.0,
        "2nd bus output",
        "Output",
    ]

    #Build the Network object, which stores all other objects
    network = pypsa.Network(override_component_attrs=override_component_attrs)

    #load graph
    nodes = pd.Index(pro_names)
    edges = pd.read_csv("data/edges.txt", sep=",", header=None)

    #set times
    network.set_snapshots(pd.date_range(config['tmin'],config['tmax'],freq=config['freq']))

    network.snapshot_weightings[:] = config['frequency']
    represented_hours = network.snapshot_weightings.sum()[0]
    Nyears= represented_hours/8760.

    costs = prepare_costs(Nyears, config)

    heat_demand, space_heat_demand, water_heat_demand, ashp_cop, gshp_cop, co2_totals = prepare_data(network)

    ds_solar = xr.open_dataset(snakemake.input.profile_solar)
    ds_onwind = xr.open_dataset(snakemake.input.profile_onwind)
    ds_offwind = xr.open_dataset(snakemake.input.profile_offwind)

    # network.heat_demand = heat_demand
    # heat_demand.to_hdf(snakemake.output.heat_demand_name, key='heat_demand', mode='w')


    pro_shapes = gpd.GeoDataFrame.from_file('data/province_shapes/CHN_adm1.shp')
    pro_shapes = pro_shapes.to_crs(4326)
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
    if config['add_solar_thermal']:
        network.add("Carrier","solar thermal")
    if config['add_PHS']:
        network.add("Carrier","PHS")
    if config['add_hydro']:
        network.add("Carrier","hydro")
    if config['add_H2_storage']:
        network.add("Carrier","H2")
    if config['add_battery_storage']:
        network.add("Carrier","battery")
    if config["heat_coupling"]:
        network.add("Carrier","heat")
        network.add("Carrier","water tanks")
    if config["retrofitting"]:
        network.add("Carrier", "retrofitting")
    if config["transport_coupling"]:
        network.add("Carrier","Li ion")

    if not isinstance(config['scenario']['co2_reduction'], tuple):

        if config['scenario']['co2_reduction'] is not None:

            co2_limit = 5.43*1e9

            # if config["transport_coupling"]:
            #     co2_limit += co2_totals['transport']
            #
            # if config["heat_coupling"]:
            #     co2_limit += co2_totals['heating']

            co2_limit *= 1 - config['scenario']['co2_reduction']

            network.add("GlobalConstraint",
                        "co2_limit",
                        type="primary_energy",
                        carrier_attribute="co2_emissions",
                        sense="<=",
                        constant=co2_limit)



    #load demand data
    with pd.HDFStore(f'data/load/load_{config["load_year"]}_weatheryears_1979_2016_TWh.h5', mode='r') as store:
        load = 1e6 * store['load'].loc[network.snapshots]

    load.columns = pro_names

    network.madd("Load", nodes, bus=nodes, p_set=load[nodes])

    #add renewables
    Onwind_p_nom = pd.read_hdf('data/p_nom/onwind_p_nom.h5')
    network.madd("Generator",
                 nodes,
                 suffix=' onwind',
                 bus=nodes,
                 carrier="onwind",
                 p_nom_extendable=True,
                 p_nom=Onwind_p_nom,
                 p_nom_min=Onwind_p_nom,
                 p_nom_max=ds_onwind['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['onwind','fixed'],
                 marginal_cost=costs.at['onwind','VOM'],
                 p_max_pu=ds_onwind['profile'].transpose('time','bus').to_pandas())

    offwind_nodes = ds_offwind['bus'].to_pandas().index
    Offwind_p_nom = pd.read_hdf('data/p_nom/offwind_p_nom.h5')
    network.madd("Generator",
                 offwind_nodes,
                 suffix=' offwind',
                 bus=offwind_nodes,
                 carrier="offwind",
                 p_nom_extendable=True,
                 p_nom=Offwind_p_nom,
                 p_nom_min=Offwind_p_nom,
                 p_nom_max=ds_offwind['p_nom_max'].to_pandas(),
                 capital_cost = costs.at['offwind','fixed'],
                 p_max_pu=ds_offwind['profile'].transpose('time','bus').to_pandas(),
                 marginal_cost=costs.at['offwind','VOM']
                 )
    
    Solar_p_nom = pd.read_hdf('data/p_nom/solar_p_nom.h5')
    
    network.madd("Generator",
                 nodes,
                 suffix=' solar',
                 bus=nodes,
                 carrier="solar",
                 p_nom_extendable=True,
                 p_nom=Solar_p_nom,
                 p_nom_min=Solar_p_nom,
                 p_nom_max=ds_solar['p_nom_max'].to_pandas(),
                 capital_cost = 0.5*(costs.at['solar-rooftop','fixed']+costs.at['solar-utility','fixed']),
                 p_max_pu=ds_solar['profile'].transpose('time','bus').to_pandas(),
                 marginal_cost=costs.at['solar','VOM']
                 )

    #add conventionals
    for generator,carrier in [("OCGT","gas")]:
        # add converter from fuel source
        
        OCGT_p_nom = pd.read_hdf('data/p_nom/OCGT_p_nom.h5')
        
        network.madd("Bus",
                     nodes,
                     suffix=" " + carrier,
                     x=pro_shapes['geometry'].centroid.x,
                     y=pro_shapes['geometry'].centroid.y,
                     carrier=carrier)

        network.madd("Link",
                     nodes,
                     suffix=" " + generator,
                     bus0=nodes + " " + carrier,
                     bus1=nodes,
                     marginal_cost=costs.at[generator,'efficiency']*costs.at[generator,'VOM'], #NB: VOM is per MWel
                     capital_cost=costs.at[generator,'efficiency']*costs.at[generator,'fixed'], #NB: fixed cost is per MWel
                     p_nom_extendable=False,
                     p_nom=OCGT_p_nom,
                     p_nom_min=OCGT_p_nom,
                     efficiency=costs.at[generator,'efficiency'])

        network.madd("Store",
                     nodes + " " + carrier + " Store",
                     bus=nodes + " " + carrier,
                     e_nom_extendable=True,
                     e_min_pu=-1.,
                     marginal_cost=costs.at[carrier,'fuel'])
        
       
    if config['add_coal']:
          network.add("Carrier","coal",co2_emissions=costs.at['coal','CO2 intensity'])
          coal_p_nom = pd.read_hdf('data/p_nom/Coal_p_nom.h5')
          network.madd("Generator",
                     nodes,
                     suffix=' coal',
                     p_nom_extendable= False,
                     p_nom= coal_p_nom,
                     p_nom_min=coal_p_nom,
                     bus=nodes,
                     carrier="coal",
                     efficiency=0.45,
                     marginal_cost=costs.at['coal','fuel'])
    

    if config['add_nuclear']:
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

    if config['add_PHS']:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        hydrocapa_df = pd.read_csv('data/hydro/PHS_p_nom.csv', index_col=0)
        phss = hydrocapa_df.index[hydrocapa_df['MW'] > 0].intersection(nodes)
        if config['hydro']['hydro_capital_cost']:
            cc=costs.at['PHS','fixed']
        else:
            cc=0.

        network.madd("StorageUnit",
                     phss,
                     suffix=" PHS",
                     bus=phss,
                     carrier="PHS",
                     p_nom_extendable=False,
                     p_nom=hydrocapa_df.loc[phss]['MW'],
                     p_nom_min=hydrocapa_df.loc[phss]['MW'],
                     max_hours=config['hydro']['PHS_max_hours'],
                     efficiency_store=np.sqrt(costs.at['PHS','efficiency']),
                     efficiency_dispatch=np.sqrt(costs.at['PHS','efficiency']),
                     cyclic_state_of_charge=True,
                     capital_cost = cc,
                     marginal_cost=0.)

    if config['add_hydro']:

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
            y=dams['geometry'].centroid.y)

        dam_buses = network.buses[network.buses.carrier=='stations']


        # ### add hydro reservoirs as stores

        initial_capacity = pd.read_pickle('data/hydro/reservoir_initial_capacity.pickle')
        effective_capacity = pd.read_pickle('data/hydro/reservoir_effective_capacity.pickle')
        initial_capacity.index = dams.index
        effective_capacity.index = dams.index
        initial_capacity = initial_capacity/water_consumption_factor
        effective_capacity=effective_capacity/water_consumption_factor


        network.madd('Store',
            dams.index,
            suffix=' reservoir',
            bus=dam_buses.index,
            e_nom=effective_capacity,
            e_initial=initial_capacity,
            e_cyclic=True,
            marginal_cost=config['costs']['marginal_cost']['hydro'])

        ### add hydro turbines to link stations to provinces
        network.madd('Link',
                    dams.index,
                    suffix=' turbines',
                    bus0=dam_buses.index,
                    bus1=dams['Province'],
                    p_nom=10 * dams['installed_capacity_10MW'],
                    efficiency= 1)
        # p_nom * efficiency = 10 * dams['installed_capacity_10MW']


        ### add rivers to link station to station
        bus0s = [0, 21, 11, 19, 22, 29, 8, 40, 25, 1, 7, 4, 10, 15, 12, 20, 26, 6, 3, 39]
        bus1s = [5, 11, 19, 22, 32, 8, 40, 25, 35, 2, 4, 10, 9, 12, 20, 23, 6, 17, 14, 16]

        for bus0, bus2 in list(zip(dams.index[bus0s], dam_buses.iloc[bus1s].index)):

            # normal flow
            network.links.at[bus0 + ' turbines', 'bus2'] = bus2
            network.links.at[bus0 + ' turbines', 'efficiency2'] = 1.

        ### spillage
        for bus0, bus1 in list(zip(dam_buses.iloc[bus0s].index, dam_buses.iloc[bus1s].index)):
            network.add('Link',
                       "{}-{}".format(bus0,bus1) + ' spillage',
                       bus0=bus0,
                       bus1=bus1,
                       p_nom_extendable=True)

        dam_ends = [dam for dam in range(len(dams.index)) if (dam in bus1s and dam not in bus0s) or (dam not in bus0s+bus1s)]

        # for bus0 in dam_buses.iloc[dam_ends].index:
        #     network.add('Link',
        #                 bus0 + ' spillage',
        #                 bus0=bus0,
        #                 bus1='Tibet',
        #                 p_nom_extendable=True,
        #                 efficiency=0.0)

        #### add inflow as generators
        # only feed into hydro stations which are the first of a cascade
        inflow_stations = [dam for dam in range(len(dams.index)) if not dam in bus1s ]

        for inflow_station in inflow_stations:

            # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow
            p_nom = (inflow.loc[pd.date_range('2016-01-01 00:00','2016-12-31 23:00',freq=config['freq'])]/water_consumption_factor).iloc[:,inflow_station].max()
            p_pu = (inflow.loc[pd.date_range('2016-01-01 00:00','2016-12-31 23:00',freq=config['freq'])]/water_consumption_factor).iloc[:,inflow_station] / p_nom
            p_pu.index = network.snapshots
            network.add('Generator',
                       dams.index[inflow_station] + ' inflow',
                       bus=dam_buses.iloc[inflow_station].name,
                       carrier='hydro_inflow',
                       p_max_pu=p_pu.clip(1.e-6),
                       # p_min_pu=p_pu.clip(1.e-6),
                       p_nom=p_nom)

            # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

        # ### add fake hydro just to introduce capital cost
        if config['add_hydro'] and config['hydro']['hydro_capital_cost']:
            hydro_cc=costs.at['hydro','fixed']

            network.madd('StorageUnit',
                        dams.index,
                        suffix=' hydro dummy',
                        bus=dams['Province'],
                        carrier='hydro',
                        p_nom=10 * dams['installed_capacity_10MW'],
                        p_max_pu=0.,
                        p_min_pu=0.,
                        capital_cost=hydro_cc)

        # else: hydro_cc=0.

    if config['add_H2_storage']:

        network.madd("Bus",
                     nodes,
                     suffix=" H2",
                     x=pro_shapes['geometry'].centroid.x,
                     y=pro_shapes['geometry'].centroid.y,
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

    if config['add_methanation']:
        network.madd("Link",
                     nodes + " Sabatier",
                     bus0=nodes+" H2",
                     bus1=nodes+" gas",
                     p_nom_extendable=True,
                     efficiency=costs.at["methanation","efficiency"],
                     capital_cost=costs.at["methanation","fixed"])

    if config['add_helmeth']:
        network.madd("Link",
                     nodes + " helmeth",
                     bus0=nodes,
                     bus1=nodes+" gas",
                     p_nom_extendable=True,
                     efficiency=costs.at["helmeth","efficiency"],
                     capital_cost=costs.at["helmeth","fixed"])

    if config['add_battery_storage']:

        network.madd("Bus",
                     nodes,
                     suffix=" battery",
                     x=pro_shapes['geometry'].centroid.x,
                     y=pro_shapes['geometry'].centroid.y,
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
                     marginal_cost=0.,
                     p_nom_extendable=True)

    #Sources:
    #[HP]: Henning, Palzer http://www.sciencedirect.com/science/article/pii/S1364032113006710
    #[B]: Budischak et al. http://www.sciencedirect.com/science/article/pii/S0378775312014759

    if config["heat_coupling"]:

        # urban = nodes

        #NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

        #central are urban nodes with district heating
        # central = nodes ^ urban

        central_fraction = pd.read_hdf("data/heating/DH_percent2020.h5")

        network.madd("Bus",
                nodes,
                suffix=" decentral heat",
                x=pro_shapes['geometry'].centroid.x,
                y=pro_shapes['geometry'].centroid.y,
                carrier="heat")

        network.madd("Bus",
                nodes,
                suffix=" central heat",
                x=pro_shapes['geometry'].centroid.x,
                y=pro_shapes['geometry'].centroid.y,
                carrier="heat")

        network.madd("Load",
                     nodes,
                     suffix=" decentral heat",
                     bus=nodes + " decentral heat",
                     p_set=heat_demand[nodes].multiply(1-central_fraction))

        network.madd("Load",
                     nodes,
                     suffix=" central heat",
                     bus=nodes + " central heat",
                     p_set=heat_demand[nodes].multiply(central_fraction))

        if config['add_heat_pumps']:

            for cat in [' decentral ', ' central ']:
                network.madd("Link",
                             nodes,
                             suffix=cat + "heat pump",
                             bus0=nodes,
                             bus1=nodes + cat + "heat",
                             efficiency=ashp_cop[nodes] if config["time_dep_hp_cop"] else costs.at[cat.lstrip()+"air-sourced heat pump",'efficiency'],
                             capital_cost=costs.at[cat.lstrip()+'air-sourced heat pump','investment'],
                             p_nom_extendable=True)

            network.madd("Link",
                         nodes,
                         suffix=" ground heat pump",
                         bus0=nodes,
                         bus1=nodes + " decentral heat",
                         efficiency=gshp_cop[nodes] if config["time_dep_hp_cop"] else costs.at['decentral ground-sourced heat pump','efficiency'],
                         capital_cost=costs.at['decentral ground-sourced heat pump','investment'],
                         p_nom_extendable=True)

        if config['retrofitting']:

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
                         p_nom_max=config['retroI-fraction']*space_peak*(1-central_fraction),
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=config['retrofitting-cost_factor']*costs.at['retrofitting I','fixed']*square_metres/(config['retroI-fraction']*space_peak))

            network.madd('Generator',
                         retro_nodes,
                         suffix=' retrofitting II',
                         bus=retro_nodes+' heat',
                         carrier="retrofitting",
                         p_nom_extendable=True,
                         p_nom_max=config['retroII-fraction']*space_peak*(1-central_fraction),
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=config['retrofitting-cost_factor']*costs.at['retrofitting II','fixed']*square_metres/(config['retroII-fraction']*space_peak))

            network.madd('Generator',
                         retro_nodes,
                         suffix=' urban retrofitting I',
                         bus=retro_nodes+' urban heat',
                         carrier="retrofitting",
                         p_nom_extendable=True,
                         p_nom_max=config['retroI-fraction']*space_peak*central_fraction,
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=config['retrofitting-cost_factor']*costs.at['retrofitting I','fixed']*square_metres/(config['retroI-fraction']*space_peak))

            network.madd('Generator',
                         retro_nodes,
                         suffix=' urban retrofitting II',
                         bus=retro_nodes+' urban heat',
                         carrier="retrofitting",
                         p_nom_extendable=True,
                         p_nom_max=config['retroII-fraction']*space_peak*central_fraction,
                         p_max_pu=space_pu,
                         p_min_pu=space_pu,
                         capital_cost=config['retrofitting-cost_factor']*costs.at['retrofitting II','fixed']*square_metres/(config['retroII-fraction']*space_peak))

        if config['add_thermal_storage']:

            for cat in [' decentral ', ' central ']:
                network.madd("Bus",
                            nodes,
                            suffix=cat + "water tanks",
                            x=pro_shapes['geometry'].centroid.x,
                            y=pro_shapes['geometry'].centroid.y,
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
                             standing_loss=1-np.exp(-1/(24.* (config["tes_tau"] if cat==' decentral ' else 180.))),  # [HP] 180 day time constant for centralised, 3 day for decentralised
                             capital_cost=costs.at[cat.lstrip()+'water tank storage','fixed']/(1.17e-3*40)) #conversion from EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K

        if config["add_boilers"]:

            for cat in [" decentral ", " central "]:
                # network.madd("Link",
                #              nodes + cat + "resistive heater",
                #              bus0=nodes,
                #              bus1=nodes + cat + "heat",
                #              efficiency=costs.at[cat.lstrip()+'resistive heater','efficiency'],
                #              capital_cost=costs.at[cat.lstrip()+'resistive heater','efficiency']*costs.at[cat.lstrip()+'resistive heater','fixed'],
                #              p_nom_extendable=True)

                network.madd("Bus",
                             nodes,
                             suffix=cat + "gas",
                             x=pro_shapes['geometry'].centroid.x,
                             y=pro_shapes['geometry'].centroid.y,
                             carrier='gas')

                network.madd("Store",
                             nodes + cat + "heat gas Store",
                             bus=nodes + cat + "gas",
                             e_nom_extendable=True,
                             e_min_pu=-1.,
                             marginal_cost=costs.at['gas','fuel'])

                network.madd("Link",
                             nodes + cat + "gas boiler",
                             p_nom_extendable=True,
                             bus0=nodes + cat + "gas",
                             bus1=nodes + cat + "heat",
                             efficiency=costs.at[cat.lstrip()+'gas boiler','efficiency'],
                             capital_cost=costs.at[cat.lstrip()+'gas boiler','efficiency']*costs.at[cat.lstrip()+'gas boiler','fixed'])

        if config["add_chp"]:
            
            # network.madd("Bus",
            #              nodes,
            #              suffix=' CHPgas',
            #              x=pro_shapes['geometry'].centroid.x,
            #              y=pro_shapes['geometry'].centroid.y,
            #              carrier='gas')
            #
            # network.madd("Store",
            #              nodes + " CHPgas Store",
            #              bus=nodes + " CHPgas",
            #              e_nom_extendable=True,
            #              e_min_pu=-1.,
            #              marginal_cost=costs.at['gas','fuel'])
            #
            # network.madd("Link",
            #              nodes,
            #              suffix=" CHPgas",
            #              bus0=nodes + " CHPgas",
            #              bus1=nodes,
            #              bus2=nodes + " central heat",
            #              p_nom_extendable=True,
            #              capital_cost=costs.at['central CHP','fixed'],
            #              efficiency=config['chp_parameters']['eff_el'],
            #              efficiency2=config['chp_parameters']['eff_th'])
            
            Coal_CHP_p_nom = pd.read_hdf('data/p_nom/CHP_p_nom.h5')
            
            network.madd("Bus",
                         nodes,
                         suffix=' CHPcoal',
                         x=pro_shapes['geometry'].centroid.x,
                         y=pro_shapes['geometry'].centroid.y,
                         carrier='coal')

            network.madd("Store",
                         nodes + " CHPcoal Store",
                         bus=nodes + " CHPcoal",
                         e_nom_extendable=True,
                         e_min_pu=-1.,
                         marginal_cost=costs.at['coal','fuel'])

            network.madd("Link",
                         nodes,
                         suffix=" CHPcoal",
                         bus0=nodes + " CHPcoal",
                         bus1=nodes,
                         bus2=nodes + " central heat",
                         p_nom_extendable=False,
                         capital_cost=10000,
                         p_nom=Coal_CHP_p_nom,
                         p_nom_min=Coal_CHP_p_nom,
                         efficiency=config['chp_parameters']['eff_el'],
                         efficiency2=config['chp_parameters']['eff_th']
                         )

        if config["add_solar_thermal"]:

            #this is the amount of heat collected in W per m^2, accounting
            #for efficiency
            with pd.HDFStore(snakemake.input.solar_thermal_name, mode='r') as store:
                #1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
                solar_thermal = config['solar_cf_correction'] * store['solar_thermal_profiles']/1e3

            for cat in [' decentral ']:
                network.madd("Generator",
                             nodes,
                             suffix=cat + "solar thermal collector",
                             bus=nodes + cat + "heat",
                             carrier="solar thermal",
                             p_nom_extendable=True,
                             capital_cost=costs.at[cat.lstrip()+'solar thermal','fixed'],
                             p_max_pu=solar_thermal[nodes].clip(1.e-4))


    if config['add_dac']: # Direct Air Capture

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

    if config["transport_coupling"]:

        network.madd("Bus",
                     nodes,
                     suffix=" EV battery",
                     x=pro_shapes['geometry'].centroid.x,
                     y=pro_shapes['geometry'].centroid.y,
                     carrier="Li ion")

        network.madd("Load",
                     nodes,
                     suffix=" transport",
                     bus=nodes + " EV battery",
                     p_set=(1-config['transport_fuel_cell_share'])*(transport[nodes]+shift_df(transport[nodes],1)+shift_df(transport[nodes],2))/3.)

        p_nom = transport_data["number cars"]*0.011*(1-config['transport_fuel_cell_share'])  #3-phase charger with 11 kW * x% of time grid-connected

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

        if config["add_v2g"]:

            network.madd("Link",
                         nodes,
                         suffix=" V2G",
                         bus1=nodes,
                         bus0=nodes + " EV battery",
                         p_nom=p_nom,
                         p_max_pu=avail_profile[nodes],
                         efficiency=0.9)  #[B]



        if config["add_bev"]:

            network.madd("Store",
                         nodes,
                         suffix=" battery storage",
                         bus=nodes + " EV battery",
                         e_cyclic=True,
                         e_nom=transport_data["number cars"]*0.05*config["bev_availability"]*(1-config['transport_fuel_cell_share']), #50 kWh battery http://www.zeit.de/mobilitaet/2014-10/auto-fahrzeug-bestand
                         e_max_pu=1,
                         e_min_pu=dsm_profile[nodes])


        if config['transport_fuel_cell_share'] != 0:

            network.madd("Load",
                         nodes,
                         suffix=" transport fuel cell",
                         bus=nodes + " H2",
                         p_set=config['transport_fuel_cell_share']/0.58*transport[nodes])

    #add lines
    if not config['no_lines']:

        lengths = 1.25 * np.array([haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                      [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]) for name0,name1 in edges.values])

        if config['line_volume_limit_max'] is not None:
            cc = Nyears*0.01 # Set line costs to ~zero because we already restrict the line volume
        else:
            cc = (config['line_cost_factor']*lengths*[HVAC_cost_curve(l) for l in lengths]) * 1.5 * 1.02 * Nyears*annuity(40.,config['costs']['discountrate'])


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
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_networks', flexibility='seperate_co2_reduction', line_limits='opt',
                                   CHP_emission_accounting='dresden', co2_reduction='0.0',opts='ll')
    configure_logging(snakemake)

    population = pd.read_hdf(snakemake.input.population_name)

    network = prepare_network(snakemake.config)

    network.export_to_netcdf(snakemake.output.network_name)
