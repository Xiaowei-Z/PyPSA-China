


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


from matplotlib.patches import Circle, Ellipse

from matplotlib.legend_handler import HandlerPatch


from scipy.stats import pearsonr


def plot_costs():

    line_limits = ["0","opt"]

    costs = pd.read_csv(snakemake.input.costs,header=[0,1,2],index_col=[0,1,2]).sort_index()
    metrics = pd.read_csv(snakemake.input.metrics,header=[0,1,2],index_col=[0]).sort_index()

    costs = costs.groupby(costs.index.get_level_values(2)).sum()

    costs = costs/1e9

    costs = costs.groupby(costs.index.map(rename_techs)).sum()

    to_drop = costs.index[costs.max(axis=1).fillna(0.) < snakemake.config['plotting']['costs_threshold']]

    print("dropping")

    print(costs.loc[to_drop])

    costs = costs.drop(to_drop)

    new_index = (preferred_order&costs.index).append(costs.index.difference(preferred_order))

    costs = costs.loc[new_index]

    fig, axes = plt.subplots(1,2,sharey=True)
    fig.set_size_inches((10,4))


    for i,ax in enumerate(axes):
        line_limit = line_limits[i]
        df = costs["all_flex-central"][line_limit]

        df = df.rename(lambda x: float(x),axis=1)

        to_plot = df.T.sort_index()

        to_plot.index = 100*to_plot.index

        print(to_plot)

        print(to_plot.sum(axis=1))

        to_plot.plot(kind="area",stacked=True,linewidth=0,ax=ax,
                     color=[snakemake.config['plotting']['tech_colors'][i] for i in df.index])

        handles,labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()
        ax.set_ylim([0,760])

        ax.set_xlim([0,50])

        ax.set_ylabel("System Cost [EUR billion per year]")

        ax.set_xlabel("CO$_2$ emitted versus 1990 [%]")

        ax.grid(axis="y")

        if line_limit == "0":
            ax.set_title("Costs with No Transmission")
        elif line_limit == "opt":
            ax.set_title("Costs with Optimal Transmission")

        ax.legend().set_visible(False)


    #framealpha stops transparency
    #bbox: first is x, second is y
    fig.legend(handles,labels,ncol=4,bbox_to_anchor=(0.99, 0.94),framealpha=1.)#loc="upper center",
    fig.tight_layout()

    fig.savefig(snakemake.output.co2_reduction,transparent=True)


def plot_price():

    line_limit = "opt"

    metrics = pd.read_csv(snakemake.input.metrics,header=[0,1,2],index_col=[0]).sort_index()

    fig, ax = plt.subplots(1,1,sharey=True)
    fig.set_size_inches((5,3.5))

    df = metrics["all_flex-central"][line_limit]

    df = df.rename(lambda x: float(x),axis=1).loc[["co2_shadow"]]

    to_plot = df.T.sort_index()

    to_plot.index = 100*to_plot.index

    to_plot.plot(linewidth=2,ax=ax)

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylabel("CO$_2$ price [EUR/tCO$_2$]")

    ax.set_xlabel("CO$_2$ emitted versus 1990 [%]")

    ax.set_xlim([0,50])

    ax.grid(axis="y")

    ax.legend().set_visible(False)

    fig.tight_layout()

    fig.savefig(snakemake.output.co2_price,transparent=True)


def plot_price_statistics():
    nice_names = {"zero_hours" : "zero-price hours [%]",
                  "mean" : "mean prices [EUR/MWh]",
                  "standard_deviation" : "standard deviation prices [EUR/MWh]"}

    price_statistics = pd.read_csv(snakemake.input.price_statistics,header=[0,1,2],index_col=[0]).sort_index()

    fig, ax = plt.subplots(1,1,sharey=True)

    fig.set_size_inches((5,3.5))

    line_limit = "opt"
    df = price_statistics["all_flex-central"][line_limit]

    df = df.rename(lambda x: float(x),axis=1)

    df.loc["zero_hours"] = 100*df.loc["zero_hours"]

    df = df.rename(nice_names)

    to_plot = df.T.sort_index()

    to_plot.index = 100*to_plot.index

    print(to_plot)

    to_plot.plot(linewidth=2,ax=ax)

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_xlabel("CO$_2$ emitted versus 1990 [%]")

    ax.set_xlim([0,50])
    ax.set_ylim([0,100])

    ax.grid(axis="y")
    fig.tight_layout()

    fig.savefig(snakemake.output.price_statistics,transparent=True)

def my_rename_techs(tech):
    if tech == "solar":
        return "solar PV"
    if tech == "onwind":
        return "onshore wind"
    if tech == "offwind":
        return "offshore wind"
    if "CHP" in tech:
        return "CHP"

    return tech

def color(tech):

    if tech in snakemake.config["plotting"]["tech_colors"]:
        return snakemake.config["plotting"]["tech_colors"][tech]
    if "H2" in tech:
        return "m"
    if "battery" in tech:
        return "slategray"

def style(tech):

    if "Electrolysis" in tech:
        return ":"
    if " charger" in tech:
        return ":"

    return "-"

def plot_market_values():
    selection = ["solar","onwind","offwind","central CHP electric","H2 Electrolysis","H2 Fuel Cell"]
    #"battery charger","battery discharger"] #"OCGT"


    line_limit = "opt"

    market_values = pd.read_csv(snakemake.input.market_values,header=[0,1,2],index_col=[0]).sort_index()

    weighted_prices = pd.read_csv(snakemake.input.weighted_prices,header=[0,1,2],index_col=[0]).sort_index()

    fig, ax = plt.subplots(1,1,sharey=True)

    fig.set_size_inches((5,3.5))

    df = market_values["all_flex-central"][line_limit]

    prices = weighted_prices["all_flex-central"][line_limit]

    df = df.divide(prices.loc["electricity"])*100.

    df = df.rename(lambda x: float(x),axis=1)

    df = df.loc[selection]

    df = df.rename(my_rename_techs)

    to_plot = df.T.sort_index()

    to_plot.index = 100*to_plot.index

    print(to_plot)

    to_plot.plot(linewidth=2,ax=ax,color=[color(i) for i in df.index],style=[style(i) for i in df.index])

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_xlabel("CO$_2$ emitted versus 1990 [%]")
    ax.set_ylabel("Market value [%]")

    ax.set_xlim([0,50])

    ax.set_ylim([0,400])

    ax.grid(axis="y")

    fig.tight_layout()

    fig.savefig(snakemake.output.market_values,transparent=True)


def plot_fossil_share():
    line_limit = "opt"

    supply_energy = pd.read_csv(snakemake.input.supply_energy,header=[0,1,2],index_col=[0,1,2]).sort_index()

    energy = pd.read_csv(snakemake.input.energy,header=[0,1,2],index_col=[0,1]).sort_index()

    fig, ax = plt.subplots(1,1)

    fig.set_size_inches((5,3.5))

    result = pd.DataFrame()

    e_df = energy["all_flex-central"][line_limit].rename(lambda x: float(x),axis=1)

    natural_gas = e_df.loc["stores","gas Store",]

    #losses are 0.4 of 0.6 produced gas
    synthetic_gas = e_df.loc["links","Sabatier"]/(-1.5)

    fraction_natural = natural_gas / (natural_gas + synthetic_gas)


    for sector in ["electricity","heat"]:

        if sector == "electricity":
            index = ["electricity"]
            from_re = pd.Index(["offwind","onwind","ror","solar","PHS","hydro"])
            from_methane = pd.Index(["OCGT","central CHP electric"])
            primary = from_re|from_methane

        else:
            index = ["heat","urban heat"]

            from_electricity = pd.Index(["central heat pump","heat pump","ground heat pump",
                                         "resistive heater","central resistive heater"])

            from_re = pd.Index(["central solar thermal collector","solar thermal collector"])

            from_methane = pd.Index(["central CHP heat","central gas boiler","gas boiler"])

            primary = from_re|from_methane|from_electricity

        df = supply_energy["all_flex-central"][line_limit].loc[index]

        df = df.groupby(level=2).sum()

        if "transmission lines" in df.index:
            df = df.drop("transmission lines")

        df = df.rename(lambda x: float(x),axis=1)

        if sector == "electricity":
            total_fossil = df.loc[from_methane].sum()*fraction_natural
        else:
            total_fossil = df.loc[from_methane].sum()*fraction_natural + result["electricity"]*df.loc[from_electricity].sum()

        result[sector] = total_fossil/df.loc[primary].sum()

    result = result*100

    result = result.sort_index()

    result.index = 100*result.index

    result.plot(linewidth=2,ax=ax)

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_xlabel("CO$_2$ emitted versus 1990 [%]")

    ax.set_ylabel("Fossil fuel share [%]")

    ax.set_xlim([0,50])

    ax.grid(axis="y")

    fig.tight_layout()

    fig.savefig(snakemake.output.fossil_share,transparent=True)




def plot_curtailment():

    selection = ["solar","offwind","onwind"]

    line_limit = "opt"

    curtailment = pd.read_csv(snakemake.input.curtailment,header=[0,1,2],index_col=[0]).sort_index()

    fig, ax = plt.subplots(1,1,sharey=True)

    fig.set_size_inches((5,3.5))

    df = curtailment["all_flex-central"][line_limit]

    df = df.rename(lambda x: float(x),axis=1)

    df = df.loc[selection]

    print(df)

    df = df.rename(my_rename_techs)

    dfT = df.T.sort_index()

    #remove offwind where no actual offwind generation
    dfT.loc[0.3:1,"offshore wind"] = np.nan

    dfT.index = 100*dfT.index

    print(dfT)

    dfT.name = None

    dfT.columns.name = None

    dfT.index.name = None

    dfT.plot(linewidth=2,ax=ax,color=[color(i) for i in df.index])

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_xlabel("CO$_2$ emitted versus 1990 [%]")

    ax.set_ylabel("Curtailment [% of available energy]")

    ax.set_xlim([0,50])

    ax.grid(axis="y")

    fig.tight_layout()

    fig.savefig(snakemake.output.curtailment,transparent=True)




def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()
    def axes2pt():
        return np.diff(ax.transData.transform([(0,0), (1,1)]), axis=0)[0] * (72./fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses: e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}



def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0,0), radius=(s/scale)**0.5, **kw) for s in sizes]


def plot_spatial_costs(flex,line_limit):

    n = pypsa.Network(snakemake.config['results_dir'] + "version-{}/postnetworks/postnetwork-{}-{}-0.h5".format(snakemake.config['version'],flex,line_limit))

    assign_groups(n)

    #Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.index.str.len() != 2],inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in ["generators","links","stores","storage_units"]:
        getattr(n,comp)["country"] = getattr(n,comp).index.str[:2]
        getattr(n,comp)["nice_group"] = getattr(n,comp).group.map(rename_techs)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs = pd.concat((costs,(getattr(n,comp).capital_cost*getattr(n,comp)[attr]).groupby((getattr(n,comp).country,getattr(n,comp).nice_group)).sum().unstack().fillna(0.)),axis=1)

    costs = costs.groupby(costs.columns,axis=1).sum()

    costs.drop(list(costs.columns[(costs == 0.).all()]) + ["transmission lines"],axis=1,inplace=True)

    new_columns = (preferred_order&costs.columns).append(costs.columns.difference(preferred_order))

    costs = costs[new_columns]

    print(costs)
    print(costs.sum())

    costs = costs.stack()#.sort_index()

    print(costs)

    fig, ax = plt.subplots(1,1)

    fig.set_size_inches(6,4.3)

    bus_size_factor =10e9
    linewidth_factor=5e3
    line_color="gray"

    n.buses.loc["NO",["x","y"]] = [9.5,61.5]


    line_widths_exp = pd.concat(dict(Line=n.lines.s_nom_opt, Link=n.links.p_nom_opt))



    n.plot(bus_sizes=costs/bus_size_factor,
           bus_colors=snakemake.config['plotting']['tech_colors'],
           line_colors=dict(Line=line_color, Link=line_color),
           line_widths=line_widths_exp/linewidth_factor,
           ax=ax)



    if line_limit != "0":

        handles = make_legend_circles_for([5e9, 1e9], scale=bus_size_factor, facecolor="gray")
        labels = ["{} bEUR/a".format(s) for s in (5, 1)]
        l2 = ax.legend(handles, labels,
                       loc="upper left", bbox_to_anchor=(0.01, 1.01),
                       labelspacing=1.0,
                       framealpha=1.,
                       title='System cost',
                       handler_map=make_handler_map_to_scale_circles_as_in(ax))
        ax.add_artist(l2)

        handles = []
        labels = []

        for s in (10, 5):
            handles.append(plt.Line2D([0],[0],color=line_color,
                                  linewidth=s*1e3/linewidth_factor))
            labels.append("{} GW".format(s))
        l1 = l1_1 = ax.legend(handles, labels,
                              loc="upper left", bbox_to_anchor=(0.24, 1.01),
                              framealpha=1,
                              labelspacing=0.8, handletextpad=1.5,
                              title='Transmission')
        ax.add_artist(l1_1)


    else:
        techs = costs.index.levels[1]
        handles = []
        labels = []
        for t in techs:
            handles.append(plt.Line2D([0], [0], color=snakemake.config['plotting']['tech_colors'][t], marker='o', markersize=8, linewidth=0))
            labels.append(t)
        l3 = ax.legend(handles, labels,
                       loc="upper left", bbox_to_anchor=(0.01, 1.01),
                       framealpha=1.,
                       handletextpad=0., columnspacing=0.5, ncol=1, title=None)

        ax.add_artist(l3)

    #ax.set_title("Scenario {} with {} transmission".format(snakemake.config['plotting']['scenario_names'][flex],"optimal" if line_limit == "opt" else "no"))


    fig.tight_layout()

    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/spatial-costs.pdf".format(snakemake.config['version']),transparent=True)



def plot_time_series(flex,line_limit):

    n = pypsa.Network(snakemake.config['results_dir'] + "version-{}/postnetworks/postnetwork-{}-{}-0.h5".format(snakemake.config['version'],flex,line_limit))

    assign_groups(n)

    suffix = ""

    n.buses["suffix"] = n.buses.index.str[2:]

    buses = n.buses.index[n.buses.suffix == suffix]


    prices = n.buses_t.marginal_price[buses].mean(axis=1)



    gas_store = (n.stores_t.p[n.stores.index[n.stores.group == "gas Store"]].sum(axis=1)/1000.)

    vre = n.generators_t.p.sum(axis=1)/1000.
    wind = n.generators_t.p.groupby(n.generators.carrier,axis=1).sum()[["offwind","onwind"]].sum(axis=1)/1000.


    load = n.loads_t.p[n.loads.index[n.buses.suffix[n.loads.bus] == suffix]].sum(axis=1)

    techs = ["BEV charger","H2 Electrolysis","resistive heater","battery charger","ground heat pump","central heat pump","central resistive heater","urban resistive heater","urban heat pump"]

    load += n.links_t.p0.groupby(n.links.group,axis=1).sum()[techs].sum(axis=1)

    print("Pearson:")

    print("load-prices",pearsonr(load,prices))
    print("vre-prices",pearsonr(vre,prices))
    print("residual load-prices",pearsonr(load-vre,prices))

    print("wind-prices",pearsonr(wind,prices))
    print("gas store,prices",pearsonr(gas_store,prices))


    fig, ax = plt.subplots(1,1,sharey=True)
    fig.set_size_inches((5,3.5))

    df = pd.DataFrame(index=n.snapshots)

    df.index.name = None

    df["gas dispatch [GW]"] = gas_store

    df["average electricity price [EUR/MWh]"] = prices

    df.plot(alpha=0.7,ax=ax,grid=True)

    ax.set_xlim([n.snapshots[0],n.snapshots[-1]])


    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/gas_v_prices.pdf".format(snakemake.config['version']),transparent=True)


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
        for item in ["costs","metrics","price_statistics","market_values","weighted_prices","supply_energy","energy","curtailment"]:
            snakemake.input[item] = snakemake.config['summary_dir'] + 'version-{version}/csvs/{item}.csv'.format(version=snakemake.config['version'],item=item)

        for item in ["co2_reduction","co2_price","price_statistics","market_values","fossil_share","curtailment"]:
            snakemake.output[item] = snakemake.config['summary_dir'] + 'version-{version}/paper_graphics/{item}.pdf'.format(version=snakemake.config['version'],item=item)


    plot_costs()

    plot_price()

    plot_price_statistics()

    plot_market_values()

    plot_fossil_share()

    plot_curtailment()

    plot_spatial_costs("all_flex-central","opt")

    plot_time_series("all_flex-central","opt")
