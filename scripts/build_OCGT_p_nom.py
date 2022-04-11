## Build OCGT capacity
## Data: 2021中国燃气发电报告
from functions import *

def csv_to_df(csv_name=None):
    
    df = pd.read_csv(csv_name, index_col=0, header=3, skiprows=[35])
    
    df = df.apply(pd.to_numeric)

    return df['MW'].reindex(pro_names)

def build_OCGT_p_nom():

    OCGT_capacity = csv_to_df(csv_name='data/p_nom/OCGT_p_nom.csv')

    OCGT_capacity.name = "OCGT_capacity"

    OCGT_capacity.to_hdf(snakemake.output.outfile, key=OCGT_capacity.name)


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(outfile="data/p_nom/OCGT_p_nom.h5")

    build_OCGT_p_nom()
