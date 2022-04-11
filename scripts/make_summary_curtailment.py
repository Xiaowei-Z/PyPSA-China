

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


def calculate_curtailment(n,label,curtailment):

    


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

outputs = ["curtailments"]

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


    dict_of_dfs = make_summaries(networks_dict)
    
    to_csv(dict_of_dfs)