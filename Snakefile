# SPDX-FileCopyrightText: : 2012 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# Build capacities in China

ATLITE_NPROCESSES = config['atlite'].get('nprocesses', 4)

rule build_p_nom:
    output:
        coal_capacity = "data/p_nom/coal_p_nom.h5"
        CHP_capacity="data/p_nom/CHP_p_nom.h5"
        OCGT_capacity="data/p_nom/OCGT_p_nom.h5"
        offwind_capacity="data/p_nom/offwind_p_nom.h5"
        onwind_capacity="data/p_nom/onwind_p_nom.h5"
        solar_capacity="data/p_nom/solar_p_nom.h5"
    threads:1
    resources: mem_mb=500
    script: "scripts/build_p_nom.py"
        
rule build_population:
    output:
        outfile="data/population/population.h5"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/build_population.py"
        
if config['enable'].get('build_cutout', False):
    rule build_cutout:
        input: 
            regions_onshore="data/resources/regions_onshore.geojson",
            regions_offshore="data/resources/regions_offshore.geojson"
        output: "cutouts/{cutout}.nc"
        log: "logs/build_cutout/{cutout}.log"
        benchmark: "benchmarks/build_cutout_{cutout}" #what is benchmark?
        threads: ATLITE_NPROCESSES
        resources: mem_mb=ATLITE_NPROCESSES * 1000
        script: "scripts/build_cutout.py"
    
   
rule build_population_gridcell_map:
    input:
        infile="data/population/population.h5"
    output:
        outfile="data/population/population_gridcell_map.h5"
    threads: 1
    resources: mem_mb=35000
    script: "scripts/build_population_gridcell_map.py"

rule build_solar_thermal_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        outfile="data/heating/solar_thermal-{angle}.h5".format(angle=config['solar_thermal_angle'])
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_solar_thermal_profiles.py"

rule build_heat_demand_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        outfile="data/heating/daily_heat_demand.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_heat_demand_profiles.py"

rule build_cop_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        outfile="data/heating/cop.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_cop_profiles.py"

rule build_temp_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        outfile="data/heating/temp.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_temp_profiles.py"

rule build_energy_totals:
    input:
        infile1="data/population/population.h5",
        infile2="data/population/population_gridcell_map.h5"
    output:
        outfile1="data/energy_totals2020.h5",
    	outfile2="data/co2_totals.h5",
    threads: 1
    resources: mem_mb=10000
    script: "scripts/build_energy_totals.py"

rule make_options:
    input:
        options_name="options.yml"
    output:
        options_name=config['results_dir'] + 'version-' + str(config['version']) + '/options/options-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}.yml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/make_options.py"

rule prepare_networks:
    input:
        options_name=config['results_dir'] + 'version-' + str(config['version']) + '/options/options-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}.yml',
        population_name="data/population/population.h5",
        solar_thermal_name="data/heating/solar_thermal-{angle}.h5".format(angle=config['solar_thermal_angle']),
    	heat_demand_name="data/heating/daily_heat_demand.h5",
    	cop_name="data/heating/cop.h5",
        energy_totals_name="data/energy_totals.h5",
        co2_totals_name="data/co2_totals.h5",
        temp="data/heating/temp.h5"
    output:
        network_name=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}.nc'
    threads: 1
    resources: mem_mb=10000
    script: "scripts/prepare_network.py"
