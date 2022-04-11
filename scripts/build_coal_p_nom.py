## Build coal power plant capacity
## Data: 中国电力统计年鉴2021 - 分地区发电装机容量（火电） - 2020年数据 [China Electricity Statistical Yearbook 2021, 2020 coal power plant capacity]
## Data: 中国电力统计年鉴2021 - 分地区6000千瓦及以上电厂供热容量 - 2020年数据 
from functions import *

def csv_to_df(csv_name=None):
    
    df = pd.read_csv(csv_name, index_col=0, header=3, skiprows=[35])
    
    df = df.apply(pd.to_numeric)

    return df['MW'].reindex(pro_names)

def build_coal_p_nom():

    coal_capacity = csv_to_df(csv_name='data/p_nom/coal_p_nom.csv') #coal power plant capacity + CHP capacity

    CHP_capacity = csv_to_df(csv_name='data/p_nom/CHP_p_nom.csv') 

    coal_capacity = coal_capacity - 0.75*CHP_capacity #coal power plant capacity 

    coal_capacity.name = "coal_capacity"

    coal_capacity.to_hdf(snakemake.output.outfile, key=coal_capacity.name)


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(outfile="data/p_nom/coal_p_nom.h5")

    build_coal_p_nom()
