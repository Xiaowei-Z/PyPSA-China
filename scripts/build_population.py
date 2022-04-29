from functions import *

def csv_to_df(csv_name=None):
    
    df = pd.read_csv(csv_name, index_col=0, header=3, skiprows=[35,36,37])
    
    df = df.apply(pd.to_numeric)

    return df['2016'].reindex(pro_names)

def build_population():

    population = 1.e3 * csv_to_df(csv_name='data/population/population_from_National_Data.csv')

    population.name = "population"

    population.to_hdf(snakemake.output.outfile, key=population.name)


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        snakemake = Dict()
        snakemake.output = Dict(outfile="data/population.h5")

    build_population()
