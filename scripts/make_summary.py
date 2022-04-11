
from six import iteritems

import sys

import pandas as pd

import pypsa

from vresutils.costdata import annuity

from prepare_network import generate_periodic_profiles

import yaml

import helper

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}

from functions import pro_names
from joblib import Parallel, parallel_backend, delayed

from multiprocessing import Manager,Pool,Lock
from functools import partial


#separator to find group name
find_by = " "

#defaults for group name
defaults = {"Load" : "electricity", "Link" : "transmission lines"}

snapshot_weightings = 3.


def assign_groups(n):

    for c in n.iterate_components(n.one_port_components|n.branch_components):

        c.df["group"] = defaults.get(c.name,"default")

        ifind = pd.Series(c.df.index.str.find(find_by),c.df.index)

        for i in ifind.value_counts().index:
            #these have already been assigned defaults
            if i == -1:
                continue

            names = ifind.index[ifind == i]

            c.df.loc[names,'group'] = names.str[i+1:]


def assign_provinces(n):

    for c in n.iterate_components(n.one_port_components|n.branch_components):

        # c.df["province"] = defaults.get(c.name,"default")

        ifind = pd.Series(c.df.index.str.find(find_by),c.df.index)
        # print(ifind)

        for i in ifind.value_counts().index:
            #these have already been assigned defaults
            # if i == -1:
                # continue

            names = ifind.index[ifind == i]
            # print(names.str[:])

            c.df.loc[names,'province'] = names.str[:i]


def calculate_costs(n,label,costs):

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        nonnegative_p_nom = c.df[opt_name.get(c.name,"p") + "_nom_opt"] - c.df[opt_name.get(c.name,"p") + "_nom"]
        nonnegative_p_nom[nonnegative_p_nom < 0] = 0
        capital_costs = c.df.capital_cost*nonnegative_p_nom
        capital_costs_grouped = capital_costs.groupby(c.df.group).sum()

        costs = costs.reindex(costs.index|pd.MultiIndex.from_product([[c.list_name],["capital"],capital_costs_grouped.index]))

        costs.loc[idx[c.list_name,"capital",list(capital_costs_grouped.index)],label] = capital_costs_grouped.values

        if c.name == "Link":
            p = c.pnl.p0.sum()
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.copy()
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings,axis=0).sum()

        marginal_costs = p*c.df.marginal_cost
        marginal_costs_grouped = marginal_costs.groupby(c.df.group).sum()

        costs = costs.reindex(costs.index|pd.MultiIndex.from_product([[c.list_name],["marginal"],marginal_costs_grouped.index]))

        costs.loc[idx[c.list_name,"marginal",list(marginal_costs_grouped.index)],label] = marginal_costs_grouped.values

    #add back in costs of links if there is a line volume limit
    if label[1] != "opt":
        costs.loc[("links-added","capital","transmission lines"),label] = ((400*1.25*n.links.length+150000.)*n.links.p_nom_opt)[n.links.group == "transmission lines"].sum()*1.5*(annuity(40., 0.07)+0.02)

    #add back in all hydro
    if options['add_hydro'] and options['hydro_capital_cost']:
        costs.loc[("storage_units","capital","hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro","p_nom"].sum()
    if options['add_PHS'] == True:
        costs.loc[("storage_units","capital","PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS","p_nom"].sum()
    if options['add_ror'] == True:
        costs.loc[("generators","capital","ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror","p_nom"].sum()

    return costs


def calculate_curtailment(n,label,curtailment):

    avail = n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt).sum().groupby(n.generators.group).sum()
    used = n.generators_t.p.sum().groupby(n.generators.group).sum()

    curtailment[label] = (((avail - used)/avail)*100).round(3)

    return curtailment


def calculate_energy(n,label,energy):

    for c in n.iterate_components(n.one_port_components|n.branch_components):

        if c.name in n.one_port_components:
            c_energies = c.pnl.p.sum().multiply(c.df.sign).groupby(c.df.group).sum()*snapshot_weightings
        else:
            c_energies = (-c.pnl.p1.sum() - c.pnl.p0.sum()).groupby(c.df.group).sum()*snapshot_weightings

        energy = energy.reindex(energy.index|pd.MultiIndex.from_product([[c.list_name],c_energies.index]))

        energy.loc[idx[c.list_name,list(c_energies.index)],label] = c_energies.values

    return energy


def calculate_supply(n,label,supply):
    """calculate the max dispatch of each component at the buses where the loads are attached"""

    load_types = n.loads.group.value_counts().index

    for i in load_types:

        buses = n.loads.bus[n.loads.group == i].values

        bus_map = pd.Series(False,index=n.buses.index)

        bus_map.loc[buses] = True

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0:
                continue

            s = c.pnl.p.max().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'group']).sum()

            supply = supply.reindex(supply.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
            supply.loc[idx[i,c.list_name,list(s.index)],label] = s.values


        for c in n.iterate_components(n.branch_components):

            for end in ["0","1"]:

                items = c.df.index[c.df["bus" + end].map(bus_map)]

                if len(items) == 0:
                    continue

                #lots of sign compensation for direction and to do maximums
                s = (-1)**(1-int(end))*((-1)**int(end)*c.pnl["p"+end][items]).max().groupby(c.df.loc[items,'group']).sum()

                supply = supply.reindex(supply.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
                supply.loc[idx[i,c.list_name,list(s.index)],label] = s.values

    return supply


def calculate_supply_energy(n,label,supply_energy):
    """calculate the total dispatch of each component at the buses where the loads are attached"""

    load_types = n.loads.group.value_counts().index

    for i in load_types:

        buses = n.loads.bus[n.loads.group == i].values

        bus_map = pd.Series(False,index=n.buses.index)

        bus_map.loc[buses] = True

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0:
                continue

            s = c.pnl.p.sum().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'group']).sum()

            supply_energy = supply_energy.reindex(supply_energy.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
            supply_energy.loc[idx[i,c.list_name,list(s.index)],label] = s.values


        for c in n.iterate_components(n.branch_components):

            for end in ["0","1"]:

                items = c.df.index[c.df["bus" + end].map(bus_map)]

                if len(items) == 0:
                    continue

                s = (-1)*c.pnl["p"+end][items].sum().groupby(c.df.loc[items,'group']).sum()

                supply_energy = supply_energy.reindex(supply_energy.index|pd.MultiIndex.from_product([[i],[c.list_name],s.index]))
                supply_energy.loc[idx[i,c.list_name,list(s.index)],label] = s.values

    return supply_energy


def calculate_metrics(n,label,metrics):

    metrics = metrics.reindex(metrics.index|pd.Index(["line_volume","line_volume_shadow","co2_shadow"]))

    metrics.at["line_volume",label] = (n.links.length*n.links.p_nom_opt)[n.links.group == "transmission lines"].sum()

    if "line_volume_limit" in n.shadow_prices.index:
        metrics.at["line_volume_shadow",label] = n.shadow_prices.at["line_volume_limit","value"]

    
    metrics.at["co2_shadow",label] = n.global_constraints.at["co2_limit","mu"]/1e3

    return metrics


def calculate_prices(n,label,prices):

    bus_type = pd.Series([s.split(' ', maxsplit=1)[-1] for s in list(n.buses.index)],n.buses.index).replace(pro_names,"electricity")

    prices = prices.reindex(prices.index|bus_type.value_counts().index)

    #WARNING: this is time-averaged, should really be load-weighted average
    prices[label] = n.buses_t.marginal_price.mean().groupby(bus_type).mean()

    return prices


def calculate_weighted_prices(n,label,weighted_prices):
    # Warning: doesn't include storage units as loads


    weighted_prices = weighted_prices.reindex(pd.Index(["electricity","heat","space heat","urban heat","space urban heat","gas","H2"]))

    link_loads = {"electricity" :  ["heat pump", "resistive heater", "battery charger", "H2 Electrolysis"],
                  "heat" : ["water tanks charger"],
                  "urban heat" : ["water tanks charger"],
                  "space heat" : [],
                  "space urban heat" : [],
                  "gas" : ["OCGT","gas boiler","CHP electric","CHP heat"],
                  "H2" : ["Sabatier", "H2 Fuel Cell"]}

    for carrier in link_loads:

        if carrier == "electricity":
            suffix = ""
        elif carrier[:5] == "space":
            suffix = carrier[5:]
        else:
            suffix =  " " + carrier

        buses = n.buses.index[n.buses.index.str[2:] == suffix]

        if buses.empty:
            continue

        if carrier in ["H2","gas"]:
            load = pd.DataFrame(index=n.snapshots,columns=buses,data=0.)
        elif carrier[:5] == "space":
            load = n.heat_demand[buses.str[:2]].rename(columns=lambda i: str(i)+suffix)
        else:
            load = n.loads_t.p_set[buses]


        for tech in link_loads[carrier]:

            names = n.links.index[n.links.index.to_series().str[-len(tech):] == tech]

            if names.empty:
                continue

            load += n.links_t.p0[names].groupby(n.links.loc[names,"bus0"],axis=1).sum(axis=1)

        #Add H2 Store when charging
        if carrier == "H2":
            stores = n.stores_t.p[buses+ " Store"].groupby(n.stores.loc[buses+ " Store","bus"],axis=1).sum(axis=1)
            stores[stores > 0.] = 0.
            load += -stores

        weighted_prices.loc[carrier,label] = (load*n.buses_t.marginal_price[buses]).sum().sum()/load.sum().sum()

        if carrier[:5] == "space":
            print(load*n.buses_t.marginal_price[buses])

    return weighted_prices


def calculate_market_values(n, label, market_values):
    # Warning: doesn't include storage units

    n.buses["suffix"] = n.buses.index.str[2:]

    suffix = ""

    buses = n.buses.index[n.buses.suffix == suffix]


    ## First do market value of generators ##

    generators = n.generators.index[n.buses.loc[n.generators.bus,"suffix"] == suffix]

    techs = n.generators.loc[generators,"carrier"].value_counts().index

    market_values = market_values.reindex(market_values.index | techs)


    for tech in techs:
        gens = generators[n.generators.loc[generators,"carrier"] == tech]

        dispatch = n.generators_t.p[gens].groupby(n.generators.loc[gens,"bus"],axis=1).sum().reindex(columns=buses,fill_value=0.)

        revenue = dispatch*n.buses_t.marginal_price[buses]

        market_values.at[tech,label] = revenue.sum().sum()/dispatch.sum().sum()



    ## Now do market value of links ##

    for i in ["0","1"]:
        all_links = n.links.index[n.buses.loc[n.links["bus"+i],"suffix"] == suffix]

        techs = n.links.loc[all_links,"group"].value_counts().index

        market_values = market_values.reindex(market_values.index | techs)

        for tech in techs:
            links = all_links[n.links.loc[all_links,"group"] == tech]

            dispatch = n.links_t["p"+i][links].groupby(n.links.loc[links,"bus"+i],axis=1).sum().reindex(columns=buses,fill_value=0.)

            revenue = dispatch*n.buses_t.marginal_price[buses]

            market_values.at[tech,label] = revenue.sum().sum()/dispatch.sum().sum()

    return market_values


def calculate_price_statistics(n, label, price_statistics):

    price_statistics = price_statistics.reindex(price_statistics.index|pd.Index(["zero_hours","mean","standard_deviation"]))

    n.buses["suffix"] = n.buses.index.str[2:]

    suffix = ""

    buses = n.buses.index[n.buses.suffix == suffix]


    threshold = 0.1 #higher than phoney marginal_cost of wind/solar

    df = pd.DataFrame(data=0.,columns=buses,index=n.snapshots)

    df[n.buses_t.marginal_price[buses] < threshold] = 1.

    price_statistics.at["zero_hours", label] = df.sum().sum()/(df.shape[0]*df.shape[1])

    price_statistics.at["mean", label] = n.buses_t.marginal_price[buses].unstack().mean()

    price_statistics.at["standard_deviation", label] = n.buses_t.marginal_price[buses].unstack().std()

    return price_statistics

def calculate_P_nom_opt(n, label, P_nom_opt):
    p_nom_opt=n.generators.p_nom_opt.groupby(n.generators.carrier,axis=0).sum()

    p_nom_opt=p_nom_opt.drop("hydro_inflow")

    OCGT_ind = n.links.index[n.links.index.str[-4:]=="OCGT"]

    OCGT_p_nom_opt=n.links.p_nom_opt[OCGT_ind].sum()

    p_nom_opt['OCGT']=OCGT_p_nom_opt

    CHPcoal_ind=n.links.index[n.links.index.str[-16:]=="CHPcoal electric"]

    CHPcoal_p_nom_opt=n.links.p_nom_opt[CHPcoal_ind].sum()

    p_nom_opt['CHP coal']=CHPcoal_p_nom_opt

    P_nom_opt[label] = p_nom_opt

    return P_nom_opt


outputs = ["costs",
           "curtailment",
           "energy",
           "supply",
           "supply_energy",
           "prices",
           "weighted_prices",
           "price_statistics",
           "market_values",
           "metrics",
           "P_nom_opt"
           ]


def calculation_parallel(dict_of_dfs, lock, label, filename):

    with lock:

        n = helper.Network(filename)

        assign_groups(n)

        for output in outputs:
            dict_of_dfs[output] = globals()["calculate_" + output](n, label, dict_of_dfs[output])



def make_summaries(networks_dict):

    columns = pd.MultiIndex.from_tuples(networks_dict.keys(),names=["scenario","line_volumn_limit","co2_reduction","CHP_emission_accounting"])

    dict_of_dfs = {}

    for output in outputs:
        dict_of_dfs[output] = pd.DataFrame(columns=columns,dtype=float)

    for label, filename in iteritems(networks_dict):
        print(label, filename)

        n = helper.Network(filename)

        assign_groups(n)

        for output in outputs:
            dict_of_dfs[output] = globals()["calculate_" + output](n, label, dict_of_dfs[output])

    # with parallel_backend('threading'):
    # Parallel(n_jobs=-1, batch_size=1, verbose=100, require='sharedmem')(delayed(calculation_parallel)(label, filename, dict_of_dfs) for label, filename in iteritems(networks_dict))

    return dict_of_dfs


def to_csv(dict_of_dfs):

    for key in dict_of_dfs:
        dict_of_dfs[key].to_csv(snakemake.output[key])


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        # snakemake.input['heat_demand_name'] = snakemake.config['results_dir'] + 'version-' + str(snakemake.config['version']) + '/prenetworks/heat_demand.h5'
        snakemake.output = Dict()
        for item in outputs:
            snakemake.output[item] = snakemake.config['summary_dir'] + 'version-{version}/csvs/{item}.csv'.format(version=snakemake.config['version'],item=item)
            
    print(outputs)

    suffix = "nc"# if int(snakemake.config['version']) > 100 else "h5"

    networks_dict = {(flexibility, line_limit, co2_reduction, CHP_emission_accounting):
                     '{results_dir}version-{version}/postnetworks/postnetwork-{flexibility}-{line_limit}-{co2_reduction}-{CHP_emission_accounting}.{suffix}'\
                     .format(results_dir=snakemake.config['results_dir'],
                             version=snakemake.config['version'],
                             flexibility=flexibility,
                             line_limit=line_limit,
                             co2_reduction=co2_reduction,
                             CHP_emission_accounting=CHP_emission_accounting,
                             suffix=suffix)\
                     for flexibility in snakemake.config['scenario']['flexibility']
                     for line_limit in snakemake.config['scenario']['line_limits']
                     for co2_reduction in snakemake.config['scenario']['co2_reduction']
                     for CHP_emission_accounting in snakemake.config['scenario']['CHP_emission_accounting']}

    options = yaml.load(open("options.yml","r"))

    # with pd.HDFStore(snakemake.input.heat_demand_name, mode='r') as store:
    #     #the ffill converts daily values into hourly values
    #     heat_demand_df = store['heat_demand']

    dict_of_dfs = make_summaries(networks_dict)

    # manager = Manager()

    # columns = pd.MultiIndex.from_tuples(networks_dict.keys(),names=["scenario","line_volume_limit","CHP_emission_accounting","co2_reduction"])

    # dict_of_dfs = manager.dict()
    # lock = manager.Lock()

    # for output in outputs:
    #     dict_of_dfs[output] = pd.DataFrame(columns=columns,dtype=float)

    # arguments = [(label, filename) for label, filename in iteritems(networks_dict)]

    # p = Pool()
    # partial_calculation_parallel = partial(calculation_parallel, dict_of_dfs, lock)
    # t = p.starmap(partial_calculation_parallel, arguments)
    # p.close()
    # p.join()

    # export to CSV
    to_csv(dict_of_dfs)
