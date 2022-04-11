import pandas as pd
import numpy as np
import os
#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pypsa
from plot_summary import rename_techs, preferred_order
from make_summary import assign_groups, assign_provinces
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
# import cartopy.crs as ccrs

from joblib import Parallel, delayed


rename_to_merge_tech = {'H2 Electrolysis': 'hydrogen',
                        'H2 Fuel Cell': 'hydrogen', 
                        'H2 Store': 'hydrogen',
                        'battery charger': 'battery',
                        'battery discharger': 'battery',
                        'central CHPgas electric': 'CHP gas',
                        'central CHPgas heat': 'CHP gas',
                        'central CHPcoal electric':'CHP coal',
                        'central CHPcoal heat': 'CHP coal',
                        'central gas boiler': 'gas boiler',
                        'decentral gas boiler': 'gas boiler',
                        'central heat pump': 'PTH',
                        'decentral heat pump': 'PTH',
                        'central resistive heater': 'PTH',
                        'decentral resistive heater': 'PTH',
                        'ground heat pump':'PTH',
                        'central solar thermal collector': 'solar thermal',
                        'decentral solar thermal collector': 'solar thermal',
                        'decentral water tanks charger': 'storage unit',
                        'decentral water tanks discharger': 'storage unit',
                        'central water tanks charger': 'storage unit',
                        'central water tanks discharger': 'storage unit',
                        'central water tank': 'storage unit',
                        'decentral water tank': 'storage unit',
                        'hydrogen': 'storage unit',
                        'battery': 'storage unit',
                        'solar': 'solar PV'
                       }

override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]


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


def aggregate_values_wrt_province(df):

    df['province'] = [ind.split()[0] for ind in df.index]

    df = df.groupby('province').sum()

    df.index.name = 'province'

    return df[0]


def plot_primary_energy(n, version, flex, line_limit, co2_reduction, CHP_emission_accounting, snakemake):

    assign_groups(n)

    #Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[[' ' in b for b in n.buses.index]],inplace=True)

    primary = pd.DataFrame(index=n.buses.index)

    s = n.stores_t.p[n.stores.index[n.stores.index.str[-9:] == "gas Store"]].multiply(n.snapshot_weightings,axis=0).sum().to_frame()
    primary['gas'] = aggregate_values_wrt_province(s)

    s = n.stores_t.p[n.stores.index[n.stores.index.str[-10:] == "coal Store"]].multiply(n.snapshot_weightings,axis=0).sum().to_frame()
    primary['coal'] = aggregate_values_wrt_province(s)

    # primary["hydroelectricity"] = n.storage_units_t.p[n.storage_units.index[n.storage_units.index.str[3:] == "hydro"]].sum().rename(lambda x : x[:2]).fillna(0.)

    n.generators["province"] = [ind.split()[0] for ind in n.generators.index]
    n.generators["nice_group"] = n.generators["group"].map(rename_techs)

    for carrier in n.generators.nice_group.value_counts().index:
        s = n.generators_t.p[n.generators.index[n.generators.nice_group == carrier]].multiply(n.snapshot_weightings,axis=0).sum().groupby(n.generators.province).sum().fillna(0.)

        if carrier in primary.columns:
            primary[carrier] += s
        else:
            primary[carrier] = s


    primary[primary < 0.] = 0.
    primary = primary.fillna(0.)
    # print(primary)
    # print(primary.sum())
    primary = primary.stack().sort_index()

    fig, ax = plt.subplots(1,1)#,subplot_kw={"projection":ccrs.PlateCarree()})

    fig.set_size_inches(10,6)

    bus_size_factor =2e8
    linewidth_factor=1e4
    line_color="m"

    # n.buses.loc["NO",["x","y"]] = [9.5,61.5]


    line_widths_exp = pd.concat(dict(Line=n.lines.s_nom_opt, Link=n.links.p_nom_opt))



    n.plot(bus_sizes=primary/bus_size_factor,
           bus_colors=snakemake.config['plotting']['tech_colors'],
           line_colors=dict(Line=line_color, Link=line_color),
           line_widths=line_widths_exp/linewidth_factor,
           ax=ax)



    if line_limit != "0":

        handles = make_legend_circles_for([1e8, 3e7], scale=bus_size_factor, facecolor="gray")
        labels = ["{} TWh".format(s) for s in (100, 30)]
        l2 = ax.legend(handles, labels,
                       loc="upper left", bbox_to_anchor=(0.01, 0.98),
                       labelspacing=.7,
                       framealpha=1.,
                       title='Primary energy',
                       handler_map=make_handler_map_to_scale_circles_as_in(ax))
        ax.add_artist(l2)

        handles = []
        labels = []

        for s in (10, 5):
            handles.append(plt.Line2D([0],[0],color=line_color,
                                  linewidth=s*1e3/linewidth_factor))
            labels.append("{} GW".format(s))
        l1 = l1_1 = ax.legend(handles, labels,
                              loc="upper left", bbox_to_anchor=(0.2, 0.98),
                              framealpha=1,
                              labelspacing=0.7, handletextpad=1.5,
                              title='Transmission')
        ax.add_artist(l1_1)

    # else:
        techs = ['coal', 'gas', 'offshore wind', 'onshore wind', 'solar PV',
       'solar thermal']
        handles = []
        labels = []
        for t in techs:
            handles.append(plt.Line2D([0], [0], color=snakemake.config['plotting']['tech_colors'][t], marker='o', markersize=9, linewidth=0))
            labels.append(t)
        l3 = ax.legend(handles, labels,
                       loc="lower left", 
                       bbox_to_anchor=(0.01, 0.01),
                       framealpha=1.,
                       handletextpad=0., 
                       columnspacing=0.7, 
                       ncol=1, 
                       title=None,
                       fontsize=12)

        ax.add_artist(l3)

    ax.set_title("Primary Energy {} with {} trans\n{} CO2 {} 2016".format(snakemake.config['plotting']['scenario_names'][flex],"optimal" if line_limit == "opt" else "no",co2_reduction, CHP_emission_accounting))
    fig.tight_layout()
    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/spatial-{}-{}-{}-{}.pdf".format(version,flex,line_limit,co2_reduction, CHP_emission_accounting),transparent=True)


def calculate_system_cost(n):

    assign_groups(n)
    assign_provinces(n)

    #Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[[' ' in b for b in n.buses.index]],inplace=True)

    p_nom = n.generators.p_nom.fillna(0)

    nonnegative_p_nom = n.generators.p_nom_opt - p_nom
    nonnegative_p_nom[nonnegative_p_nom < 0] = 0

    n.generators['total_capital'] = nonnegative_p_nom * n.generators.capital_cost

    capital_1 = n.generators.set_index(['province', 'group'])['total_capital']\
                .rename(rename_to_merge_tech)\
                .groupby(['province', 'group']).sum()
    
    link_nonnegative_p_nom = n.links.p_nom_opt - n.links.p_nom
    link_nonnegative_p_nom[link_nonnegative_p_nom < 0] = 0

    n.links['total_capital'] = link_nonnegative_p_nom * n.links.capital_cost
    links_costs = n.links.set_index(['province', 'group'])['total_capital']

    capital_2 = links_costs.drop(level='group', labels='transmission lines')\
                .rename(rename_to_merge_tech)\
                .groupby(['province', 'group']).sum()

    n.stores['total_capital'] = n.stores.e_nom_opt * n.stores.capital_cost
    capital_3 = n.stores.set_index(['province', 'group'])['total_capital']\
                .replace(0,np.nan).dropna().replace(np.nan,0)\
                .rename(rename_to_merge_tech)\
                .groupby(['province', 'group']).sum()

    snapshot_weightings = 3.0

    #CHP_gas_marginal
    CHP_gas_Stores_ind = n.stores.index[n.stores.index.str[-12:] == "CHPgas Store"]
    CHP_s = (n.stores_t.p[CHP_gas_Stores_ind] * n.stores.marginal_cost[CHP_gas_Stores_ind]).sum().to_frame()
    CHP_gas_costs = aggregate_values_wrt_province(CHP_s)
    CHP_gas_costs = CHP_gas_costs.to_frame().reset_index()
    CHP_gas_costs['group'] = 'gas'
    CHP_gas_costs = CHP_gas_costs.set_index(['province', 'group'])[0]
    CHP_gas_costs = CHP_gas_costs * snapshot_weightings

    #heating sector gas marginal
    heat_gas_Stores_ind = n.stores.index[n.stores.index.str[-14:] == "heat gas Store"]
    heat_s = (n.stores_t.p[heat_gas_Stores_ind] * n.stores.marginal_cost[heat_gas_Stores_ind]).sum().to_frame()
    heat_gas_costs = aggregate_values_wrt_province(heat_s)
    heat_gas_costs = heat_gas_costs.to_frame().reset_index()
    heat_gas_costs['group'] = 'gas'
    heat_gas_costs = heat_gas_costs.set_index(['province', 'group'])[0]
    heat_gas_costs = heat_gas_costs * snapshot_weightings

    #marginal_gas
    gas_Stores_ind = n.stores.index[n.stores.index.str[-9:] == "gas Store"]
    s = (n.stores_t.p[gas_Stores_ind] * n.stores.marginal_cost[gas_Stores_ind]).sum().to_frame()
    gas_costs = aggregate_values_wrt_province(s)
    gas_costs = gas_costs.to_frame().reset_index()
    gas_costs['group'] = 'gas'
    gas_costs = gas_costs.set_index(['province', 'group'])[0]
    gas_costs = gas_costs*snapshot_weightings
    gas_costs = gas_costs - CHP_gas_costs - heat_gas_costs

    #marginal_coal
    Coal_Stores_ind = n.generators.index[n.generators.index.str[-4:] == "coal"]
    Coal_s = (n.generators_t.p[Coal_Stores_ind] * n.generators.marginal_cost[Coal_Stores_ind]).sum().to_frame()
    Coal_costs = aggregate_values_wrt_province(Coal_s)
    Coal_costs = Coal_costs.to_frame().reset_index()
    Coal_costs['group'] = 'coal'
    Coal_costs = Coal_costs.set_index(['province', 'group'])[0]
    Coal_costs = Coal_costs*snapshot_weightings

    #marginal_CHPcoal
    CHP_coal_Stores_ind = n.stores.index[n.stores.index.str[-13:] == "CHPcoal Store"]
    CHP_coal_s = (n.stores_t.p[CHP_coal_Stores_ind] * n.stores.marginal_cost[CHP_coal_Stores_ind]).sum().to_frame()
    CHP_coal_costs = aggregate_values_wrt_province(CHP_coal_s)
    CHP_coal_costs = CHP_coal_costs.to_frame().reset_index()
    CHP_coal_costs['group'] = 'CHP coal'
    CHP_coal_costs = CHP_coal_costs.set_index(['province', 'group'])[0]
    CHP_coal_costs = CHP_coal_costs * snapshot_weightings

    #Modify group
    CHP_gas_costs = CHP_gas_costs.to_frame().reset_index()
    CHP_gas_costs['group'] = 'CHP gas'
    CHP_gas_costs = CHP_gas_costs.set_index(['province', 'group'])[0]
    heat_gas_costs = heat_gas_costs.to_frame().reset_index()
    heat_gas_costs['group'] = 'gas boiler'
    heat_gas_costs = heat_gas_costs.set_index(['province', 'group'])[0]
    gas_costs = gas_costs.to_frame().reset_index()
    gas_costs['group'] = 'OCGT'
    gas_costs = gas_costs.set_index(['province', 'group'])[0]
    
    todrop_capital_1 = capital_1.loc[:,['inflow']]
    capital_1 = capital_1.drop(todrop_capital_1.index)

    capital_2 = capital_2.drop(['Ahai','Baihetan','Changzhou','Dongfeng','Dongjing','Ertan','Gezhouba','Gongboxia','Gongguoqiao','Gongzui','Goupitan','Guandi','Guangzhao','Jinanqiao','Jinghong','Jinping1','Jinping2','Jishixia','Laxiwa','Lijiaxia','Longkaikou','Longyangxia','Lubuge','Ludila','Luding','Manwan','Nuozhadu','Pubugou','Ruilijiang1','Sanbanxi','Sanxia',
                'Shatuo','Shenxigou','Shuibuya','Silin','Tianshengqiao2','Tongjiezi','Wudongde','Wujiangdu','Wuqiangxi','Xiangjiaba','Xiaowan','Xiluodu'])

    total_costs = pd.concat([capital_1, capital_2, capital_3, gas_costs, CHP_gas_costs, heat_gas_costs, Coal_costs, CHP_coal_costs]).groupby(['province', 'group']).sum().unstack().stack()
    total_costs = total_costs.rename(rename_to_merge_tech, axis='index', level='group')

    total_costs = total_costs.groupby(['province', 'group']).sum()

    return total_costs    


def plot_system_cost(n, version, flex, line_limit, co2_reduction, CHP_emission_accounting, snakemake):

    total_costs = calculate_system_cost(n)

    fig, ax = plt.subplots(1,1)#,subplot_kw={"projection":ccrs.PlateCarree()})

    fig.set_size_inches(10.8,7)

    bus_size_factor =2e10
    linewidth_factor=1e4
    line_color="m"

    # n.buses.loc["NO",["x","y"]] = [9.5,61.5]


    line_widths_exp = pd.concat(dict(Line=n.lines.s_nom_opt, Link=n.links.p_nom_opt))



    n.plot(bus_sizes=total_costs/bus_size_factor,
           bus_colors=snakemake.config['plotting']['tech_colors'],
           line_colors=dict(Line=line_color, Link=line_color),
           line_widths=line_widths_exp/linewidth_factor,
           ax=ax)



    if line_limit != "0":

        handles = make_legend_circles_for([5e9, 1e9], 
            scale=bus_size_factor, 
            facecolor="gray")
        labels = ["{} billion EUR".format(s) for s in (5, 1)]
        l2 = ax.legend(handles, labels,
                       loc="upper left", 
                       bbox_to_anchor=(0.01, 0.98),
                       labelspacing=0.7,
                       framealpha=1.,
                       title='Nodal costs',
                       handler_map=make_handler_map_to_scale_circles_as_in(ax))
        ax.add_artist(l2)

        handles = []
        labels = []

        for s in (10, 5):
            handles.append(plt.Line2D([0],[0],color=line_color,
                                  linewidth=s*1e3/linewidth_factor))
            labels.append("{} GW".format(s))
        l1 = l1_1 = ax.legend(handles, labels,
                              loc="upper left", 
                              bbox_to_anchor=(0.2, 0.98),
                              framealpha=1,
                              labelspacing=0.7, 
                              handletextpad=1.5,
                              title='Transmission')
        ax.add_artist(l1_1)


    # else:
        techs = total_costs.index.levels[1]
        handles = []
        labels = []
        for t in techs:
            handles.append(plt.Line2D([0], [0],
                color=snakemake.config['plotting']['tech_colors'][t],\
                marker='o', 
                markersize=8, 
                linewidth=0))
            labels.append(t)

        l3 = ax.legend(handles, labels,
                       loc="lower left", 
                       bbox_to_anchor=(0.01, 0.01),
                       framealpha=1.,
                       handletextpad=0., 
                       columnspacing=0.5, 
                       ncol=1, 
                       title=None,
                       fontsize=12)

        ax.add_artist(l3)

    ax.set_title("Costs {} with {} trans\n{} CO2 {} 2016".format(snakemake.config['plotting']['scenario_names'][flex],"optimal" if line_limit == "opt" else "no",co2_reduction,CHP_emission_accounting))


    fig.tight_layout()

    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/spatial-costs-{}-{}-{}-{}.pdf".format(version,flex,line_limit,co2_reduction,CHP_emission_accounting),transparent=True)

def plot_system_curtailment(n, version, flex, line_limit, co2_reduction, CHP_emission_accounting, snakemake):
    
    n.buses.drop(n.buses.index[[' ' in b for b in n.buses.index]],inplace=True)
    curtailment= pd.DataFrame(index=n.buses.index)

    avail = n.generators_t.p_max_pu.multiply(n.generators['p_nom_opt']).sum()
    used = n.generators_t.p.sum()
    curtailment_all = (((avail - used)/avail)*100).round(3)

    s=curtailment_all[n.generators.index[n.generators.index.str[-6:]=='onwind']].to_frame()
    curtailment['onshore wind']=aggregate_values_wrt_province(s)

    s = curtailment_all[n.generators.index[n.generators.index.str[-7:]=='offwind']].to_frame()
    curtailment['offshore wind']=aggregate_values_wrt_province(s)

    s = curtailment_all[n.generators.index[n.generators.index.str[-5:]=='solar']].to_frame()
    curtailment['solar PV']=aggregate_values_wrt_province(s)

    s = curtailment_all[n.generators.index[n.generators.index.str[-9:]=='collector']].to_frame()
    curtailment['solar thermal']=aggregate_values_wrt_province(s)

    curtailment = curtailment.fillna(0)
    curtailment = curtailment.stack().sort_index()

    fig, ax = plt.subplots(1,1)#,subplot_kw={"projection":ccrs.PlateCarree()})

    fig.set_size_inches(10,6)
    
    line_color = 'm'
    
    bus_size_factor = 50

    n.plot(bus_sizes=curtailment/bus_size_factor,
           bus_colors=snakemake.config['plotting']['tech_colors'],
           line_colors=dict(Line=line_color, Link=line_color),
           line_widths=1.0,
           ax=ax)

    handles = make_legend_circles_for([10, 5], scale=bus_size_factor, facecolor="gray")
    labels = ["10","5"]
    l2 = ax.legend(handles, labels,
                    loc="upper left", bbox_to_anchor=(0.01, 0.98),
                    labelspacing=.7,
                    framealpha=1.,
                    title='Curtailment Percentage',
                    handler_map=make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l2)

    handles = []
    labels = []

    techs = ['offshore wind', 'onshore wind', 'solar PV',
    'solar thermal']
    for t in techs:
        handles.append(plt.Line2D([0], [0], color=snakemake.config['plotting']['tech_colors'][t], marker='o', markersize=9, linewidth=0))
        labels.append(t)
    l3 = ax.legend(handles, labels,
                    loc="lower left", 
                    bbox_to_anchor=(0.01, 0.01),
                    framealpha=1.,
                    handletextpad=0., 
                    columnspacing=0.7, 
                    ncol=1, 
                    title=None,
                    fontsize=12)

    ax.add_artist(l3)

    ax.set_title("Curtailment percentage {} with {} trans\n{} CO2 {} 2016".format(snakemake.config['plotting']['scenario_names'][flex],"optimal" if line_limit == "opt" else "no",co2_reduction, CHP_emission_accounting))
    fig.tight_layout()
    fig.savefig(snakemake.config['summary_dir'] +  "version-{}/paper_graphics/spatial-curtailment-{}-{}-{}-{}.pdf".format(version,flex,line_limit,co2_reduction, CHP_emission_accounting),transparent=True)



def load_network_and_plot_both(version, flex, line_limit, co2_reduction, CHP_emission_accounting, snakemake):

    file_name = snakemake.config['results_dir'] + 'version-{version}/postnetworks/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}.nc'.format(
                    version=version,
                    flexibility=flex,
                    line_limits=line_limit,
                    co2_reduction=co2_reduction,
                    CHP_emission_accounting=CHP_emission_accounting)

    n = pypsa.Network(file_name,override_component_attrs=override_component_attrs)

    #plot_primary_energy(n, version, flex, line_limit, co2_reduction, CHP_emission_accounting, snakemake)

    #plot_system_cost(n, version, flex, line_limit, co2_reduction, CHP_emission_accounting, snakemake)

    plot_system_curtailment(n, version, flex, line_limit, co2_reduction, CHP_emission_accounting, snakemake)



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
    
    # [load_network_and_plot_both(snakemake.config['version'], flexibility, line_limit, co2_reduction, CHP_emission_accounting, snakemake)
    #                     for flexibility in snakemake.config['scenario']['flexibility']
    #                     for line_limit in snakemake.config['scenario']['line_limits']
    #                     for co2_reduction in snakemake.config['scenario']['co2_reduction']
    #                     for CHP_emission_accounting in snakemake.config['scenario']['CHP_emission_accounting']]

    # Parallel(n_jobs=-1)(delayed(load_network_and_plot_both)(snakemake.config['version'], flexibility, line_limit, co2_reduction, CHP_emission_accounting, snakemake)
    #     for flexibility in snakemake.config['scenario']['flexibility']
    #     for line_limit in snakemake.config['scenario']['line_limits']
    #     for co2_reduction in snakemake.config['scenario']['co2_reduction']
    #     for CHP_emission_accounting in snakemake.config['scenario']['CHP_emission_accounting'])

    [load_network_and_plot_both(snakemake.config['version'], flexibility, line_limit, co2_reduction, CHP_emission_accounting, snakemake)
                        for flexibility in snakemake.config['scenario']['flexibility']
                        for line_limit in snakemake.config['scenario']['line_limits']
                        for co2_reduction in snakemake.config['scenario']['co2_reduction']
                        for CHP_emission_accounting in snakemake.config['scenario']['CHP_emission_accounting']]