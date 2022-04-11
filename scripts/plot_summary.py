
import pandas as pd

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#consolidate and rename
def rename_techs(label):
    if label[:8] == "central ":
        label = label[8:]
    if label[:10] == "decentral ":
        label = label[10:]
    if label[:6] == "urban ":
        label = label[6:]

    if "retrofitting" in label:
        label = "building retrofitting"
    if "H2" in label:
        label = "hydrogen storage"
    if "water tank" in label:
        label = "water tanks"
    if label== "water tanks":
        label = "hot water storage"
#     if "gas" in label and label != "gas boiler":
#         label = "natural gas"
    if label == "gas Store":
        label = "OCGT"
    if "heat gas Store" in label:
        label = "gas boiler"
    if "solar thermal" in label:
        label = "solar thermal"
    if label == "solar":
        label = "solar PV"
    if label == "heat pump":
        label = "air heat pump"
    if label == "Sabatier":
        label = "methanation"
    if label == "offwind":
        label = "offshore wind"
    if label == "onwind":
        label = "onshore wind"
    if label == "ror":
        label = "hydroelectricity"
    if label == "hydro":
        label = "hydroelectricity"
    if label == "PHS":
        label = "hydroelectricity"
    if label == "co2 Store":
        label = "DAC"
    if label == "coal":
        label = "coal power plant"
    if "battery" in label:
        label = "battery storage"
    if "CHPgas" in label:
        label = "CHP gas"
    if "CHPcoal" in label:
        label = "CHP coal"

    return label


preferred_order = pd.Index(["CHP coal", "CHP gas", "coal power plant", "OCGT", "onshore wind", "offshore wind", "solar PV", "gas boiler", 
                            #"air heat pump", "ground heat pump", "resistive heater",
                            "battery storage", "hydrogen storage", 
                            #"hot water storage", 
                            "transimission lines"])

def plot_costs():


    cost_df = pd.read_csv(snakemake.input.costs,index_col=list(range(3)),header=[0,1,2,3])

    # cost_df.loc[(cost_df==0).all(axis=1), :]

    # cost_df.loc[(cost_df==0).all(axis=1), :]

    # cost_df = cost_df.xs(
    #             ('seperate_co2_reduction', 
    #              'opt',
    #              'dresden'),
    #             level=[0,1,3],
    #             drop_level=True,
    #             axis=1,
    #             ).filter(
    #                 like=f"({el_emission_reduction} ", 
    #                 axis=1)

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    #convert to billions
    df = df/1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < snakemake.config['plotting']['costs_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    new_columns = df.sum().sort_values().index

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,title="scenarios without PTH under 2016 loads",stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])

    handles,labels = ax.get_legend_handles_labels()

    ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_title("scenarios without PTH under 2016 loads", fontsize=16)

    ax.set_ylabel("System Cost [EUR billion per year]",fontsize=13)

    ax.set_xlabel("CO2 reduction",fontsize=13)

    ax.set_xticklabels(['-100','0.0','0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75',
    #'0.8','0.85','0.9','0.95','1.0'
    ])

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")

    fig.tight_layout()

    fig.savefig(snakemake.output.costs,transparent=True)


def plot_energy():

    energy_df = pd.read_csv(snakemake.input.energy,index_col=list(range(2)),header=[0,1,2,3])

    # energy_df.loc[(energy_df==0).all(axis=1), :]

    # energy_df.loc[(energy_df==0).all(axis=1), :]
    
    # energy_df = energy_df.xs(
    #             ('seperate_co2_reduction', 
    #              'opt',
    #              'dresden'),
    #             level=[0,1,3],
    #             drop_level=True,
    #             axis=1,
    #             ).filter(
    #                 like=f"({el_emission_reduction} ", 
    #                 axis=1)

    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    #convert MWh to TWh
    df = df/1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.abs().max(axis=1) < snakemake.config['plotting']['energy_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    df = df.drop(index = ['station spillage','turbines'])

    reverse_index = ['battery storage','hydrogen storage', 
                     #'air heat pump','ground heat pump', 'hot water storage'
                     ]

    for i in reverse_index :
        df.loc[i] = df.loc[i] * -1

    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    new_columns = df.columns.sort_values()

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,title="scenarios without PTH under 2016 loads",stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])

    handles,labels = ax.get_legend_handles_labels()

    ax.set_ylim([snakemake.config['plotting']['energy_min'],snakemake.config['plotting']['energy_max']])

    ax.set_ylabel("Energy [TWh/a]",fontsize=13)

    ax.set_xlabel("co2 reduction",fontsize=13)

    ax.set_title("scenarios without PTH under 2016 loads", fontsize=16)

    ax.set_xticklabels(['0.0','0.05','0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75',
    #'0.8','0.85','0.9','0.95','1.0',
    '-100'])

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")


    fig.tight_layout()

    fig.savefig(snakemake.output.energy,transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()

        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)

        snakemake.input = Dict(
            costs=snakemake.config['summary_dir'] + 'version-{version}/csvs/costs.csv'.format(version=snakemake.config['version']),
            energy=snakemake.config['summary_dir'] + 'version-{version}/csvs/energy.csv'.format(version=snakemake.config['version']))
        snakemake.output = Dict(
            costs=snakemake.config['summary_dir'] + 'version-{version}/graphs/costs.pdf'.format(version=snakemake.config['version']), 
            energy=snakemake.config['summary_dir'] + 'version-{version}/graphs/energy.pdf'.format(version=snakemake.config['version']))

    # el_emission_reduction = [0.0,0.5,0.7,0.9]

    plot_costs()

    plot_energy()
