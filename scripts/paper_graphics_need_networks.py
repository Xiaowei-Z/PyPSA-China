


import pandas as pd

import numpy as np

import os

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pypsa

from plot_summary import rename_techs, preferred_order

from make_summary import assign_groups

from vresutils.costdata import annuity

def heat_demand():

    fn = os.path.join(snakemake.config['results_dir'],"version-"+str(snakemake.config['version']),"postnetworks","postnetwork-base_0.h5")

    n = pypsa.Network(fn)

    fig,ax = plt.subplots(1,1)

    fig.set_size_inches((6,4))

    s = n.loads_t.p_set.loc[:,n.loads.index[n.loads.index.str[-4:] == "heat"]].sum(axis=1)
    s /=1e6

    ax.plot(s.values,linewidth=2)

    ax.grid()

    ax.set_xlabel("Month of year")

    ax.set_xlim([0,24*365])

    ax.set_xticks(np.arange(0,24*365,24*30.5))

    ax.set_xticklabels(range(1,13))

    ax.set_ylabel("Total European heat demand [TW$_{th}$]")

    fig.tight_layout()

    fig.savefig(snakemake.output.heat_demand,transparent=True)


def plot_cold_week(flex,line_limit,bus_type,ct="EU"):
    file_name = snakemake.config['results_dir'] \
                + 'version-{version}/postnetworks/postnetwork-{flexibility}_{line_limits}.h5'.format(version=snakemake.config["version"],
                                                                                                     flexibility=flex,
                                                                                                     line_limits=line_limit)
    n = pypsa.Network(file_name)

    assign_groups(n)

    ncol = 3

    if bus_type == "":
        ylabel = "Electricity generator power [GW]"
        title = "Electricity generation in {} for scenario {}".format("Europe" if ct == "EU" else ct,
                                                                      snakemake.config['plotting']['scenario_names'][flex])
        ylim = [0,2000]
        if ct == "DE":
            ylim = [0,350]
    elif bus_type == " urban heat":
        ylabel = "High-density heat supply [GW]"
        title = "High-density heat supply in {} for scenario {}".format("Europe" if ct == "EU" else ct,
                                                                      snakemake.config['plotting']['scenario_names'][flex])
        if flex == "base":
            ylim = [0,650]
            if ct == "DE":
                ylim = [0,160]
        else:
            ylim = [-500,900]
            if ct == "DE":
                ylim = [-40,170]
                ncol=2

    bus_mask = pd.Series(n.buses.index.str[2:]==bus_type,n.buses.index)
    if ct != "EU":
        ct_mask = (n.buses.index.str[:2] == "DE")
        bus_mask = bus_mask&ct_mask

    supply = pd.DataFrame(index=n.snapshots)

    for i in range(2):
        supply = pd.concat((supply,(-1)*n.links_t["p"+str(i)].loc[:,n.links.index[n.links["bus" + str(i)].map(bus_mask)]].groupby(n.links.group,axis=1).sum()),axis=1)

    for c in n.iterate_components(pypsa.components.one_port_components ^ {"Load"}):
        supply = pd.concat((supply,c.pnl["p"].loc[:,c.df.index[c.df.bus.map(bus_mask)]].groupby(c.df.group,axis=1).sum()),axis=1)

    demand = n.loads_t["p"].loc[:,n.loads.index[n.loads.bus.map(bus_mask)]].groupby(n.loads.group,axis=1).sum()


    fig,ax = plt.subplots(1,1)

    fig.set_size_inches(6,4)

    s="2011-01-30"

    e="2011-02-05"

    if "PHS" in supply.columns:
        supply = supply.drop("PHS",axis=1)

    #supply.iloc[s:e].loc[:,(supply > -0.001).all()].plot(kind="area",legend=True,stacked=True,ax=ax)

    supply = supply.groupby(supply.columns.map(rename_techs),axis=1).sum()

    if "hot water storage" in supply.columns:
        supply["hot water charging"] = supply["hot water storage"][supply["hot water storage"] < 0.]
        supply["hot water discharging"] = supply["hot water storage"][supply["hot water storage"] > 0.]
        supply = supply.drop("hot water storage",axis=1)

    supply = supply/1e3

    if bus_type == "":
        to_drop = ["transmission lines","ground heat pump","air heat pump","resistive heater","BEV charger","battery storage","hydrogen storage"]
    elif "heat" in bus_type and "building retrofitting" in supply.columns:
        to_drop = ["building retrofitting"]
    else:
        to_drop = []


    supply = supply.drop(to_drop,axis=1)

    to_drop = supply.columns[supply.abs().max().fillna(0.) < 2.]

    print("dropping")

    print(to_drop)

    supply = supply.drop(to_drop,axis=1)


    new_index = (preferred_order&supply.columns).append(supply.columns.difference(preferred_order))

    supply = supply[new_index]

    supply.loc[s:e].plot(kind="area",legend=True,
                         stacked=True,ax=ax,
                         color=[snakemake.config['plotting']['tech_colors'][i] for i in supply.columns],
                         linewidth=0)



    #demand.sum(axis=1).loc[s:e].plot(linewidth=1,label="demand")

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim(ylim)

    ax.set_ylabel(ylabel)

    ax.set_xlabel("")

    ax.set_title(title)

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=ncol,loc="upper left",framealpha=1)
    fig.tight_layout()

    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/cold_week-{}-{}{}-{}.pdf".format(snakemake.config['version'],flex,line_limit,bus_type.replace(" ","_"),ct),transparent=True)



def plot_v2g_soc(flex,line_limit,ct):
    file_name = snakemake.config['results_dir'] \
                + 'version-{version}/postnetworks/postnetwork-{flexibility}_{line_limits}.h5'.format(version=snakemake.config["version"],
                                                                                                     flexibility=flex,
                                                                                                     line_limits=line_limit)
    n = pypsa.Network(file_name)

    s="2011-06-01"

    e="2011-06-07"

    v2g = n.stores_t.e.loc[s:e,ct + " battery storage"]

    #add back in other 50% of storage
    v2g += n.stores.loc[ct + " battery storage","e_nom"]

    print(ct,"has total BEV capacity",2*n.stores.loc[ct + " battery storage","e_nom"]/1e3)

    v2g = v2g/1e3


    fig,ax = plt.subplots(1,1)

    fig.set_size_inches(6,4)
    v2g.plot(ax=ax,linewidth=2)

    ax.set_ylim([0,2*n.stores.loc[ct + " battery storage","e_nom"]/1e3+100])


    ax.set_ylabel("Total BEV state of charge in {} [GWh]".format(ct))

    ax.set_xlabel("")

    ax.grid(True)

    #ax.legend(handles,labels,ncol=4,loc="upper left",framealpha=1)
    fig.tight_layout()

    fig.savefig(snakemake.output["v2g_soc_"+ct],transparent=True)


def central_tes_stats():
    flex = "central-tes"
    line_limit ="0"

    file_name = snakemake.config['results_dir'] \
                + 'version-{version}/postnetworks/postnetwork-{flexibility}_{line_limits}.h5'.format(version=snakemake.config["version"],
                                                                                                     flexibility=flex,
                                                                                                     line_limits=line_limit)
    n = pypsa.Network(file_name)


    central = n.stores.index[n.stores.index.str.contains("central")].str[:2]

    central_max = n.loads_t.p_set[central + " urban heat"].max().sum()

    print("central-tes peak district heating demand is",central_max,"MW")

    district_cost = n.loads_t.p_set[central + " urban heat"].max().sum()*4e5*(0.01+annuity(50,0.07))/1e9

    print("annualised cost of district is",district_cost,"billion EUR")

    #convert with 40K * 0.00117 MWh/m^3/K
    volume = n.stores.loc[central + " central water tank","e_nom_opt"].sum()/0.0468

    print("central-tes has LTES with volume",volume/1e9,"billion cubic metres")

    print("central-tes has LTES with",volume/(0.446*529e6),"cubic metres per citizen")

    flex = "tes"
    line_limit ="0"

    file_name = snakemake.config['results_dir'] \
                + 'version-{version}/postnetworks/postnetwork-{flexibility}_{line_limits}.h5'.format(version=snakemake.config["version"],
                                                                                                     flexibility=flex,
                                                                                                     line_limits=line_limit)
    n = pypsa.Network(file_name)

    tanks = n.stores.index[n.stores.bus.map(n.buses.carrier) == "water tanks"]
    #convert with 40K * 0.00117 MWh/m^3/K
    volume =  n.stores.loc[tanks,"e_nom_opt"].sum()/0.0468

    print("tes has STES with volume",volume/1e6,"million cubic metres")

    print("tes has STES with",volume/529e6,"cubic metres per citizen")

def scales():


    fig,ax = plt.subplots(1,1)

    fig.set_size_inches(5,3.5)


    file_name = snakemake.config['results_dir'] \
                + 'version-{version}/postnetworks/postnetwork-all_flex-central_0.h5'.format(version=snakemake.config["version"])

    n = pypsa.Network(file_name)

    assign_groups(n)

    e = n.stores_t.e.groupby(n.stores.group,axis=1).sum()

    e_pu = e.copy()

    minus = {"gas Store"}

    e_pu[e.columns^minus] = e_pu[e.columns^minus]/e_pu[e.columns^minus].max()

    e_pu[list(minus)] = 1-e_pu[list(minus)]/e_pu[list(minus)].min()

    nice_names = {"gas Store": "methane storage",
                  "H2 Store" : "hydrogen storage",
                  "central water tank" : "long-term hot water storage"}
    e_pu[["central water tank","H2 Store","gas Store",]].rename(nice_names,
                                                                axis=1).plot(ax=ax,
                                                                             linewidth=2,
                                                                             color=["k","m","orange"])

    ax.set_ylabel("State of charge [per unit of max]")
    ax.set_xlabel("")

    ax.set_ylim([0,1.15])

    ax.grid()

    ax.set_xlim([n.snapshots[0],n.snapshots[-1]])

    ax.legend()

    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = labels[0].replace("\n2011","")
    ax.set_xticklabels(labels)

    fig.tight_layout()

    fig.savefig(snakemake.output.scales,transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.output = Dict()
        snakemake.output.heat_demand = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/heat_demand.pdf'.format(version=snakemake.config['version'])
        snakemake.output.scales = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/scales.pdf'.format(version=snakemake.config['version'])

    scales()

    heat_demand()

    plot_cold_week("base","0","")
    plot_cold_week("base","0"," urban heat")

    plot_cold_week("central-tes","0","")
    plot_cold_week("central-tes","0"," urban heat")

    plot_cold_week("base","0","","DE")
    plot_cold_week("base","0"," urban heat","DE")

    plot_cold_week("central-tes","0","","DE")
    plot_cold_week("central-tes","0"," urban heat","DE")

    plot_v2g_soc("v2g","0","DE")

    plot_v2g_soc("v2g","0","IT")

    central_tes_stats()
