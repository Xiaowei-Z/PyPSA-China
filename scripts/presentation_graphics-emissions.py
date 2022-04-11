import pandas as pd

idx = pd.IndexSlice


#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import root


#translations for sectors
e = pd.Series()
e["public electricity and heat"] = '1.A.1.a - Public Electricity and Heat Production'
e['residential heating'] = '1.A.4.b - Residential'
e['services heating'] = '1.A.4.a - Commercial/Institutional'
e['rail transport'] = "1.A.3.c - Railways"
e["road transport"] = '1.A.3.b - Road Transportation'
e["domestic navigation"] = "1.A.3.d - Domestic Navigation"
e['international navigation'] = '1.D.1.b - International Navigation'
e["domestic aviation"] = '1.A.3.a - Domestic Aviation'
e["international aviation"] = '1.D.1.a - International Aviation'
e['agriculture'] = '3 - Agriculture'
e['waste management'] = '5 - Waste management'


def read_emissions_file():
#https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-13

    df = pd.read_csv(snakemake.input.emissions, sep="\t")

    df.loc[df["Year"] == "1985-1987","Year"] = 1986

    df["Year"] = df["Year"].astype(int)

    df = df.set_index(['Country_code', 'Pollutant_name', 'Year', 'Sector_name']).sort_index()

    return df


def plot_emissions_pie(df, pollutant, year, ct):

    pol = "All greenhouse gases - (CO2 equivalent)" if pollutant == "GHG" else pollutant

    e_emissions = pd.Series(index=list(e.index[:5]) + ["navigation","aviation"] + list(e.index[5:]))

    for k,v in e.iteritems():
        e_emissions[k] = df.loc[idx[ct,pol,year,v],"emissions"].sum()


    e_emissions["industry (non-electric)"] = df.loc[idx[ct,pol,year,
                                                        "Total (with LULUCF, without indirect CO2)"],"emissions"].sum() \
                                                        - e_emissions.sum()

    for sec in ["aviation", "navigation"]:
        secs = ["domestic " + sec,"international " + sec]
        e_emissions.loc[sec] = e_emissions.loc[secs].sum()
        e_emissions.drop(secs,inplace=True)

    if pol == "CO2":
        to_agg = ["waste management","agriculture"]
        e_emissions.loc["other"] = e_emissions.loc[to_agg].sum()
        e_emissions.drop(to_agg,inplace=True)

    print(ct,pollutant,year,e_emissions.sum())

    #cf https://www.eea.europa.eu/data-and-maps/daviz/change-of-co2-eq-emissions-2#tab-dashboard-01

    fig, ax = plt.subplots(1,1)

    fig.set_size_inches(4,4)

    ax.pie(e_emissions.values, labels=e_emissions.index, autopct='%1.1f%%')

    #fig.tight_layout()


    place = "EU28" if ct == "EUC" else ct

    fig.savefig("{}-emissions_pie-{}-{}.pdf".format(place,year,pollutant),
                bbox_inches="tight",
                transparent=True)




def plot_emissions_curve(df, pollutant="CO2", year_zero=2050, ct="EUC", budget=48, steepness=0.1):


    pol = "All greenhouse gases - (CO2 equivalent)" if pollutant == "GHG" else pollutant
    selection = "Total (with LULUCF, without indirect CO2)"

    #cf http://ec.europa.eu/eurostat/statistics-explained/index.php/Greenhouse_gas_emission_statistics_-_emission_inventories

    s = df.loc[idx[ct,pol,:,selection],"emissions"]

    s.index = s.index.droplevel([0,1,3])

    s = s/1e3

    #end of historical record, start of projection
    x0 = s.index[-1]

    x1 = year_zero

    start = s.values[-1]

    #this varies steepness/drama
    k = steepness

    interval = 0.2
    x = np.arange(x0,x1,interval)

    #find midpoint of sigmoid
    def f(parameters):
        mid = parameters[0]
        y = start*(1/(1+np.exp(k*(x-mid)))-1/(1+np.exp(k*(x1-mid))))/(1/(1+np.exp(k*(x0-mid)))-1/(1+np.exp(k*(x1-mid))))

        print(y.sum()*interval)
        return budget*1e3 - y.sum()*interval

    results = root(f,[2025])

    print("success?",results["success"])

    mid = results["x"][0]
    y = start*(1/(1+np.exp(k*(x-mid)))-1/(1+np.exp(k*(x1-mid))))/(1/(1+np.exp(k*(x0-mid)))-1/(1+np.exp(k*(x1-mid))))

    fig,ax = plt.subplots(1,1)

    fig.set_size_inches((6,4))

    s_add = pd.concat((s,pd.Series(y,x)))

    s_add = 100*s_add/s_add.max()

    s_add.plot(ax=ax,linewidth=2)

    ax.set_xlim([1990,2050])

    ax.grid()

    ax.set_ylim([0,100])

    if pol == "CO2":
        ax.set_ylabel("Fraction of 1990 CO$_2$ emissions [%]")
    else:
        ax.set_ylabel("Fraction of 1990 GHG emissions [%]")

    ax.set_xlabel("Year")

    if pol == "CO2":
        ax.set_title("EU28 budget of {} Gt CO$_2$ from 2015".format(budget))
    else:
        ax.set_title("EU28 budget of 60 Gt CO$_2$-equivalent from 2015")

    fig.tight_layout()

    fig.savefig("EU28-{}-budget-{}.pdf".format(budget,pollutant),transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.input["emissions"] = "/home/tom/store/energy/UNFCCC_v20.csv"


    df = read_emissions_file()

    for pol in ["GHG","CO2"]:
        for year in [1990,2015]:
            plot_emissions_pie(df, pol, year, "EUC")

    plot_emissions_curve(df, pollutant="CO2", year_zero=2050, ct="EUC", budget=48, steepness=0.1)

    plot_emissions_curve(df, pollutant="GHG", year_zero=2050, ct="EUC", budget=60, steepness=0.1)
