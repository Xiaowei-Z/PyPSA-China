## Build CHP capacity
## Data: 中国电力统计年鉴2021 - 分地区6000千瓦及以上电厂供热容量 - 2020年数据 

from functions import *

def csv_to_df(csv_name=None):
    
    df = pd.read_csv(csv_name, index_col=0, header=3, skiprows=[35])
    
    df = df.apply(pd.to_numeric)

    return df['MW'].reindex(pro_names)

def build_CHP_p_nom():

    CHP_capacity = csv_to_df(csv_name='data/p_nom/CHP_p_nom.csv') 

    CHP_capacity.name = "CHP_capacity"

    CHP_capacity.to_hdf(snakemake.output.outfile, key=CHP_capacity.name)


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(outfile="data/p_nom/CHP_p_nom.h5")

    build_CHP_p_nom()
