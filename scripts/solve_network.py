# -*- coding: utf-8 -*-
import datetime
import os
import sys
import warnings
from functools import partial
from math import asin, cos, radians, sin, sqrt

import helper
import numpy as np
import pandas as pd
import pyproj
import pypsa
import pytz
import yaml
from prepare_network import prepare_costs
from pyomo.environ import Constraint
from shapely.ops import transform
from six import iteritems, iterkeys, itervalues
from vresutils import timer

# from pyutilib.services import TempfileManager
# TempfileManager.tempdir = '/home/201402677/tmp'


def extra_functionality(network, snapshots):
    # add a very small penalty to (one hour of) the state of charge of
    # non-extendable storage units -- this ensures that the storage is empty in
    # (at least) one hour
    if not hasattr(network, "epsilon"):
        network.epsilon = 1e-5
    fix_sus = network.storage_units[~network.storage_units.p_nom_extendable]
    network.model.objective.expr += sum(
        network.epsilon * network.model.state_of_charge[su, network.snapshots[0]]
        for su in fix_sus.index
    )

    if options["line_volume_limit_factor"] is not None:
        # branches = network.branches()
        # extendable_branches = branches[branches.s_nom_extendable]
        network.model.line_volume_limit = pypsa.opt.Constraint(
            expr=sum(
                network.model.link_p_nom[link] * network.links.at[link, "length"]
                for link in network.links.index
                if link[2:3] == "-"
            )
            <= options["line_volume_limit_factor"] * options["line_volume_limit_max"]
        )

    if options["abs_flow_cost"]:
        controllable_branches = network.controllable_branches()
        network.model.controllable_branch_p_pos = pypsa.opf.Var(
            list(controllable_branches.index),
            network.snapshots,
            domain=pypsa.opf.NonNegativeReals,
        )
        network.model.controllable_branch_p_neg = pypsa.opf.Var(
            list(controllable_branches.index),
            network.snapshots,
            domain=pypsa.opf.NonNegativeReals,
        )

        def cb_p_pos_neg(model, cb_type, cb_name, snapshot):
            return (
                model.controllable_branch_p[cb_type, cb_name, snapshot]
                - model.controllable_branch_p_pos[cb_type, cb_name, snapshot]
                + model.controllable_branch_p_neg[cb_type, cb_name, snapshot]
                == 0
            )

        network.model.controllable_branch_p_pos_neg = pypsa.opt.Constraint(
            list(controllable_branches.index), network.snapshots, rule=cb_p_pos_neg
        )

        # \epsilon * (f_pos + f_neg) = \epsilon * abs(Flow)
        from pyomo.environ import summation

        abs_flow = summation(network.model.controllable_branch_p_pos) + summation(
            network.model.controllable_branch_p_neg
        )
        abs_flow._coef = [options["abs_flow_cost"]] * len(abs_flow._coef)

        network.model.objective.expr += abs_flow

    if options["heterogeneity"] is not None:
        # min/max own shares
        # own_carriers = ['wind','solar','hydro','OCGT']
        own_carriers = np.append(network.generators.carrier.unique(), "hydro")
        # own_carriers = ['wind','solar','hydro']
        own_gens = network.generators[network.generators.carrier.isin(own_carriers)]
        own_su = network.storage_units[network.storage_units.carrier.isin(own_carriers)]

        # heterogeneity is controlled by parameter k_own
        # min and max shares of own generation in each node n:
        # 1/k * L_n <= G^R_n <= k * L_n
        # in units of the total load L_n in n.
        k_own = options["heterogeneity"]
        if hasattr(k_own, "__len__"):
            f_lo, f_up = k_own
        else:
            f_lo, f_up = 1.0 / k_own, k_own
        factor_own = pd.DataFrame([f_lo, f_up])
        L_tot = network.loads_t.p_set.sum(axis=0)
        own_bounds = pd.DataFrame(L_tot).dot(
            factor_own.T
        )  # pandas way of getting [lower,upper]_n
        own_bounds = own_bounds.where((pd.notnull(own_bounds)), None)

        p_own = {(bus): [[], "><", own_bounds.loc[bus]] for bus in network.buses.index}

        for gen in own_gens.index:
            bus = own_gens.bus[gen]
            sign = own_gens.sign[gen]
            for sn in network.snapshots:
                p_own[(bus)][0].append((sign, network.model.generator_p[gen, sn]))

        for su in own_su.index:
            bus = own_su.bus[su]
            sign = own_su.sign[su]
            for sn in network.snapshots:
                p_own[(bus)][0].append((sign, network.model.storage_p_dispatch[su, sn]))

        pypsa.opt.l_constraint(
            network.model, "heterogeneity", p_own, network.buses.index
        )

    if options["add_battery_storage"]:

        nodes = list(network.buses.index[network.buses.carrier == "battery"])

        def battery(model, node):
            return (
                model.link_p_nom[node + " charger"]
                == model.link_p_nom[node + " discharger"]
                * network.links.at[node + " charger", "efficiency"]
            )

        network.model.battery = Constraint(nodes, rule=battery)

    # def calculate_el_and_th_fractions(network, s, sn, gc):

    #     CHP_name = s.split()[0] + " central CHP"

    #     CHP_el = network.model.link_p[CHP_name+" electric",sn] * network.links.loc[CHP_name+" electric"].efficiency
    #     CHP_th = network.model.link_p[CHP_name+" heat",sn] * network.links.loc[CHP_name+" heat"].efficiency
    #     CHP_gas_fuel = network.model.store_p[s,sn]

    #     if options['CHP_emission_accounting']=='energy':

    #         el_frac = CHP_el / (CHP_th + CHP_el)
    #         th_frac = 1 - el_frac

    #     elif options['CHP_emission_accounting']=='exergy':

    #         Ts = 85
    #         Tr = 45
    #         Th = (Ts-Tr) / np.log10(Ts/Tr)

    #         th_frac = CHP_th * (1 - network.temperature / Th) / (CHP_el + CHP_th * (1 - network.temperature / Th) )
    #         el_frac = 1 - th_frac

    #     elif options['CHP_emission_accounting']=='powerbonus':

    #         f1 = 1.05 # https://www.sciencedirect.com/science/article/pii/S0196890418308446#b0105
    #         f2 = 2.42

    #         th_frac = (CHP_gas_fuel * f1 - CHP_el * f2) / (CHP_gas_fuel * f1)
    #         el_frac = 1 - th_frac

    #     elif options['CHP_emission_accounting']=='heatbonus':

    #         f1 = 1.05
    #         f3 = 1. # dummy value

    #         el_frac = (CHP_gas_fuel * f1 - CHP_th * f3) / (CHP_gas_fuel * f1)
    #         th_frac = 1 - el_frac

    #     elif options['CHP_emission_accounting']=='alternative':

    #         eta_th = costs.at['central gas boiler','efficiency']
    #         eta_el = options['chp_parameters']['eta_elec']

    #         el_frac = (CHP_el/eta_el) / (CHP_el/eta_el + CHP_th/eta_th)
    #         th_frac = 1 - el_frac

    #     elif options['CHP_emission_accounting']=='marginal':

    #         duals = pd.Series(list(network.model.dual.values()), index=pd.Index(list(network.model.dual.keys())))
    #         marginal_price = pd.Series(list(network.model.power_balance.values()),
    #                                   index=pd.MultiIndex.from_tuples(list(network.model.power_balance.keys()))).map(duals)
    #         el_buses = network.buses.index[[~(' ' in b) for b in network.buses.index]]
    #         th_buses = network.buses.index[['central heat' in b for b in network.buses.index]]

    #         el_frac = (CHP_el * marginal_price.loc[el_buses]) / (CHP_el * marginal_price.loc[el_buses] + CHP_th * marginal_price.loc[th_buses])
    #         th_frac = 1 - el_frac

    #     else:

    #         el_frac = 1.
    #         th_frac = 0.

    #         return el_frac if gc=='co2_limit_el' else th_frac

    def define_co2_el_and_th_constraints(network, snapshots):

        el_gas_stores = network.stores.index[
            ([("gas Store" in s) & ("heat" not in s) for s in network.stores.index])
            & (~network.stores.e_cyclic)
        ]
        heat_gas_stores = network.stores.index[
            (["heat gas Store" in s for s in network.stores.index])
            & (~network.stores.e_cyclic)
        ]

        CHP_gas_stores = network.stores.index[
            (["CHPgas Store" in s for s in network.stores.index])
            & (~network.stores.e_cyclic)
        ]

        co2_el_and_th_constraints = {}
        co2_el_and_th_constraints_list = ["co2_limit_el", "co2_limit_th"]

        gas_stores = {
            co2_el_and_th_constraints_list[0]: el_gas_stores,
            co2_el_and_th_constraints_list[1]: heat_gas_stores,
        }

        for gc in co2_el_and_th_constraints_list:

            c = pypsa.opt.LConstraint(sense="<=")
            c.rhs.constant = co2_limits[gc]
            carrier_attribute = "co2_emissions"

            for carrier in network.carriers.index:
                attribute = network.carriers.at[carrier, carrier_attribute]
                # this attribute is co2_emissions=costs.at['gas','CO2 intensity']
                if attribute == 0.0:
                    continue

                c.lhs.variables.extend(
                    [
                        (-attribute, network.model.store_e[store, snapshots[-1]])
                        for store in gas_stores[gc]
                    ]
                )
                c.lhs.constant += sum(
                    attribute * network.stores.at[store, "e_initial"]
                    for store in gas_stores[gc]
                )
                # c.lhs.variables.extend([(attribute*calculate_el_and_th_fractions(network, s, sn, gc), network.model.store_p[s,sn]) for s in CHP_gas_stores for sn in snapshots])
                if options["CHP_emission_accounting"] == "200percent":

                    if gc == "co2_limit_th":
                        # 0.5 * Pth
                        c.lhs.variables.extend(
                            [
                                (
                                    attribute
                                    * 0.5
                                    * options["chp_parameters"]["eta_elec"]
                                    / options["chp_parameters"]["c_v"],
                                    network.model.link_p[
                                        s.split()[0] + " central CHP heat", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                    elif gc == "co2_limit_el":
                        # Fuel - Pth
                        c.lhs.variables.extend(
                            [
                                (attribute, network.model.store_p[s, sn])
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                        c.lhs.variables.extend(
                            [
                                (
                                    -attribute
                                    * 0.5
                                    * options["chp_parameters"]["eta_elec"]
                                    / options["chp_parameters"]["c_v"],
                                    network.model.link_p[
                                        s.split()[0] + " central CHP heat", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )

                if options["CHP_emission_accounting"] == "dresden":

                    if gc == "co2_limit_el":
                        # electricity / eta_electric; the eta_electric cancels out
                        c.lhs.variables.extend(
                            [
                                (
                                    attribute,
                                    network.model.link_p[
                                        s.split()[0] + " central CHP electric", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                    elif gc == "co2_limit_th":
                        # fuel - electricity / eta_electric; the eta_electric cancels out
                        c.lhs.variables.extend(
                            [
                                (attribute, network.model.store_p[s, sn])
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                        c.lhs.variables.extend(
                            [
                                (
                                    -attribute,
                                    network.model.link_p[
                                        s.split()[0] + " central CHP electric", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )

                if options["CHP_emission_accounting"] == "substitution":

                    eta_gas_boiler = 0.9
                    if gc == "co2_limit_el":
                        # E/2/eta_el - Q/2/eta_boiler + F/2; the eta_electric cancels out
                        c.lhs.variables.extend(
                            [
                                (
                                    attribute * 0.5,
                                    network.model.link_p[
                                        s.split()[0] + " central CHP electric", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                        c.lhs.variables.extend(
                            [
                                (
                                    -attribute
                                    * 0.5
                                    / (
                                        eta_gas_boiler
                                        * options["chp_parameters"]["eta_elec"]
                                        / options["chp_parameters"]["c_v"]
                                    ),
                                    network.model.link_p[
                                        s.split()[0] + " central CHP heat", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                        c.lhs.variables.extend(
                            [
                                (0.5 * attribute, network.model.store_p[s, sn])
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                    elif gc == "co2_limit_th":
                        # Q/2/eta_boiler - E/2/eta_el + F/2; the eta_electric cancels out
                        c.lhs.variables.extend(
                            [
                                (
                                    -attribute * 0.5,
                                    network.model.link_p[
                                        s.split()[0] + " central CHP electric", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                        c.lhs.variables.extend(
                            [
                                (
                                    attribute
                                    * 0.5
                                    / (
                                        eta_gas_boiler
                                        * options["chp_parameters"]["eta_elec"]
                                        / options["chp_parameters"]["c_v"]
                                    ),
                                    network.model.link_p[
                                        s.split()[0] + " central CHP heat", sn
                                    ],
                                )
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )
                        c.lhs.variables.extend(
                            [
                                (0.5 * attribute, network.model.store_p[s, sn])
                                for sn in snapshots
                                for s in CHP_gas_stores
                            ]
                        )

            co2_el_and_th_constraints[gc] = c

        pypsa.opt.l_constraint(
            network.model,
            "co2_el_and_th_constraints",
            co2_el_and_th_constraints,
            co2_el_and_th_constraints_list,
        )

    if isinstance(options["co2_reduction"], tuple):

        co2_limit_el = network.co2_totals["electricity"]  # already annualized

        if options["transport_coupling"]:
            co2_limit_tr += network.co2_totals["transport"]

        if options["heat_coupling"]:
            co2_limit_th = network.co2_totals["heating"]

        co2_limit_el *= 1 - options["co2_reduction"][0]
        co2_limit_th *= 1 - options["co2_reduction"][1]

        co2_limits = {"co2_limit_el": co2_limit_el, "co2_limit_th": co2_limit_th}

        define_co2_el_and_th_constraints(network, snapshots)


def solve_model(network):

    solver_name = options["solver_name"]
    solver_io = options["solver_io"]
    solver_options = options["solver_options"]
    check_logfile_option(solver_name, solver_options)
    with timer("lopf optimization"):  # as tdic['lopf_opt']:
        network.lopf(
            network.snapshots,
            solver_name=solver_name,
            solver_io=solver_io,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            keep_files=options["opf_keep_files"],
            formulation=options["formulation"],
        )

    try:
        _tn = [dd.name for dd in pypsa.opf.tdic.values()]
        _tt = [dd.time for dd in pypsa.opf.tdic.values()]
        network.timed = pd.Series(_tt, index=_tn, name=network.snapshots.size)
    except AttributeError:
        print("no timer (tdic) in opf")
        pass

    # save the shadow prices of some constraints

    def get_shadows(constraint, multiind=True):
        if len(constraint) == 0:
            return pd.Series()

        index = list(constraint.keys())
        if multiind:
            index = pd.MultiIndex.from_tuples(index)
        cdata = pd.Series(list(constraint.values()), index=index)
        return cdata.map(duals)

    network.shadow_prices = pd.DataFrame()

    if options["line_volume_limit_factor"] is not None:
        network.shadow_prices["line_volume_limit"] = network.model.dual[
            getattr(network.model, "line_volume_limit")
        ]

    if not isinstance(options["co2_reduction"], tuple):
        if options["co2_reduction"] is not None:
            network.shadow_prices["co2_constraint"] = network.global_constraints.loc[
                "co2_limit", "mu"
            ]
    else:
        duals = pd.Series(
            list(network.model.dual.values()),
            index=pd.Index(list(network.model.dual.keys())),
        )
        network.shadow_prices["co2_constraint"] = -get_shadows(
            network.model.co2_el_and_th_constraints, multiind=False
        )

    return network

    if hasattr(network, "timed"):
        network.timed.to_csv(os.path.join(results_folder_name, "times.csv"))
    # reading back:
    # timed=pd.read_csv('tt.csv',index_col=0,header=None,squeeze=True)


def check_logfile_option(solver_name, solver_options):
    # make sure to use right keyword for each solver
    #'logfile' for gurobi
    #'log' for glpk
    if "logfile" in solver_options and solver_name == "glpk":
        solver_options["log"] = solver_options.pop("logfile")
    elif "log" in solver_options and solver_name == "gurobi":
        solver_options["logfile"] = solver_options.pop("log")


def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


if __name__ == "__main__":

    options = yaml.load(open(snakemake.input.options_name, "r"))

    network = helper.Network(snakemake.input.network_name)

    # network = pypsa.Network(snakemake.input.network_name)

    # annualize co2_totals
    represented_hours = network.snapshot_weightings.sum()
    Nyears = represented_hours / 8760.0
    network.co2_totals = pd.DataFrame()
    with pd.HDFStore(snakemake.input.co2_totals_name, mode="r") as store:
        network.co2_totals = 1.0e-3 * Nyears * store["co2"]

    with pd.HDFStore(snakemake.input.temp, mode="r") as store:
        network.temperature = store["temperature"].loc[network.snapshots, :]

    network.consistency_check()

    solve_model(network)

    network.export_to_netcdf(snakemake.output.network_name)
