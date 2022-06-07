import yaml, sys
from ast import literal_eval 

input_name = snakemake.input.options_name

output_name = snakemake.output.options_name


options = yaml.load(open(input_name,'r'),Loader=yaml.Loader)


if snakemake.wildcards.line_limits == "opt":
    options['line_volume_limit_factor'] = None
else:
    options['line_volume_limit_factor'] = float(snakemake.wildcards.line_limits)


if '(' in snakemake.wildcards.co2_reduction:

    options['co2_reduction'] = literal_eval(snakemake.wildcards.co2_reduction.replace(' ', ',').replace('minus', '-'))

else:

    if snakemake.wildcards.co2_reduction != 'None':
        if snakemake.wildcards.co2_reduction[:5] == "minus":
            options['co2_reduction'] = -float(snakemake.wildcards.co2_reduction[5:])
        else:
            options['co2_reduction'] = float(snakemake.wildcards.co2_reduction)
    else:
        options['co2_reduction'] = None


flex = snakemake.wildcards.flexibility
options['flexibility'] = flex

CHP_emission_accounting = snakemake.wildcards.CHP_emission_accounting
options['CHP_emission_accounting'] = CHP_emission_accounting


def extract_fraction(flex,prefix="bev",default=0.):
    """Converts "fc" to default, "fc50" to 0.5"""
    i = flex.find(prefix)
    if i + len(prefix) == len(flex):
        return default
    else:
        return float(flex[flex.find(prefix)+len(prefix):])/100

if flex == "elec_only":
    pass
elif flex == "transport":
    options['transport_coupling'] = True
elif "bev" in flex:
    options['transport_coupling'] = True
    options['bev'] = True
    options['bev_availability'] = extract_fraction(flex,"bev",0.5)
elif "v2g" in flex:
    options['transport_coupling'] = True
    options['bev'] = True
    options['v2g'] = True
    options['bev_availability'] = extract_fraction(flex,"v2g",0.5)
elif "fc" in flex:
    options['transport_coupling'] = True
    options['transport_fuel_cell_share'] =  extract_fraction(flex,"fc",0.5)
elif flex == "base":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
elif flex=="methanation":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
elif flex == "central":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['central'] = True
elif flex == "tes":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['tes'] = True
elif flex == "central-tes":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['tes'] = True
    options['central'] = True
elif flex == "all_flex":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['tes'] = True
    options['bev'] = True
    options['v2g'] = True
elif flex == "all_flex-central":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['tes'] = True
    options['bev'] = True
    options['v2g'] = True
    options['central'] = True
elif flex == "helmeth":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['tes'] = True
    options['bev'] = True
    options['v2g'] = True
    options['central'] = True
    options['helmeth'] = True
elif flex == "dac":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['tes'] = True
    options['bev'] = True
    options['v2g'] = True
    options['central'] = True
    options['helmeth'] = True
    options['dac'] = True
elif flex in ["el_and_heating"]:
    options['transport_coupling'] = False
    options['heat_coupling'] = True
    options['chp'] = True
    options['add_methanation'] = False
    options['tes'] = False
    options['bev'] = False
    options['v2g'] = False
    options['helmeth'] = False
    options['dac'] = False
elif flex in ['seperate_co2_reduction']:
    options['chp'] = True
    options['transport_coupling'] = False
    options['heat_coupling'] = True
    options['add_methanation'] = False
    options['tes'] = False
    options['bev'] = False
    options['v2g'] = False
    options['helmeth'] = False
    options['dac'] = False
elif flex in ['seperate_co2_reduction_tes']:
    options['chp'] = True
    options['transport_coupling'] = False
    options['heat_coupling'] = True
    options['add_methanation'] = False
    options['tes'] = False
    options['bev'] = False
    options['v2g'] = False
    options['helmeth'] = False
    options['dac'] = False
elif flex == "nuclear":
    options['transport_coupling'] = True
    options['heat_coupling'] = True
    options['add_methanation'] = True
    options['tes'] = True
    options['bev'] = True
    options['v2g'] = True
    options['central'] = True
    options['helmeth'] = True
    options['nuclear'] = True
else:
    print("flexibility option",flex,"not recognised!")
    sys.exit()

#if options['heat_coupling']:
#    options['retrofitting'] = True
#options['retrofitting'] = True if 'retro' in flex else False

yaml.dump(options,open(output_name,"w"))
