# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8

import pandas as pd
from functions import pro_names

def csv_to_df(csv_name=None):
    
    df = pd.read_csv(csv_name, index_col=0, header=0)
    
    df = df.apply(pd.to_numeric)

    return df['MW'].reindex(pro_names)
    
def build_p_nom():

    coal_capacity = csv_to_df(csv_name="data/p_nom/coal_p_nom.csv") 

    CHP_capacity = csv_to_df(csv_name="data/p_nom/CHP_p_nom.csv")
    
    OCGT_capacity = csv_to_df(csv_name="data/p_nom/OCGT_p_nom.csv")

    coal_capacity.name = "coal_capacity"
    
    CHP_capacity.name = "CHP_capacity"
    
    OCGT_capacity.name = "OCGT_capacity"

    coal_capacity.to_hdf(snakemake.output.coal_capacity, key=coal_capacity.name)
    
    CHP_capacity.to_hdf(snakemake.output.CHP_capacity, key=CHP_capacity.name)
    
    OCGT_capacity.to_hdf(snakemake.output.OCGT_capacity, key=OCGT_capacity.name)
    
if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(coal_capacity="data/p_nom/coal_p_nom.h5", CHP_capacity="data/p_nom/CHP_p_nom.h5", OCGT_capacity="data/p_nom/OCGT_p_nom.h5")
        
    build_p_nom()
