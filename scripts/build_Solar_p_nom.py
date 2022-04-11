## Build Solar capacity
## Data: 中国电力统计年鉴2021 - 分地区发电装机容量（太阳能发电） - 2020年数据 

from functions import *

def csv_to_df(csv_name=None):
    
    df = pd.read_csv(csv_name, index_col=0, header=3, skiprows=[35])
    
    df = df.apply(pd.to_numeric)

    return df['MW'].reindex(pro_names)

def build_Solar_p_nom():

    Solar_capacity = csv_to_df(csv_name='data/p_nom/Solar_p_nom.csv')

    Solar_capacity.name = "Solar_capacity"

    Solar_capacity.to_hdf(snakemake.output.outfile, key=Solar_capacity.name)


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(outfile="data/p_nom/Solar_p_nom.h5")

    build_Solar_p_nom()
