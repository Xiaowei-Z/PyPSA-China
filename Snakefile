# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# Pathway to reduce CO2 emissions from 2020 to 2060

from os.path import normpath, exists
from shutil import copyfile, move

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

configfile: "config.yaml"

COSTS="data/costs.csv"
ATLITE_NPROCESSES = config['atlite'].get('nprocesses', 4)

wildcard_constraints:
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*"

rule prepare_all_networks:
    input:
        expand(config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opt}.nc',
            version=config['version'],
            **config['scenario'])

rule solve_all_networks:
    input:
        expand(config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opt}.nc',
            version=config['version'],
            **config['scenario'])

rule build_p_nom:
    output:
        coal_capacity="data/p_nom/coal_p_nom.h5",
        CHP_capacity="data/p_nom/CHP_p_nom.h5",
        OCGT_capacity="data/p_nom/OCGT_p_nom.h5",
        offwind_capacity="data/p_nom/offwind_p_nom.h5",
        onwind_capacity="data/p_nom/onwind_p_nom.h5",
        solar_capacity="data/p_nom/solar_p_nom.h5",
        nuclear_capacity="data/p_nom/nuclear_p_nom.h5"
    threads:1
    resources: mem_mb=500
    script: "scripts/build_p_nom.py"

rule build_population:
    output:
        population="data/population/population.h5"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/build_population.py"

if config['enable'].get('retrieve_cutout', True):
    rule retrieve_cutout:
        input: HTTP.remote("zenodo.org/record/6510859/files/China-2020.nc", keep_local=True, static=True)
        output: "cutouts/{cutout}.nc"
        run: move(input[0], output[0])


if config['enable'].get('build_cutout', False):
    rule build_cutout:
        input:
            regions_onshore="data/resources/regions_onshore.geojson",
            regions_offshore="data/resources/regions_offshore.geojson"
        output: "cutouts/{cutout}.nc"
        log: "logs/build_cutout/{cutout}.log"
        benchmark: "benchmarks/build_cutout_{cutout}"
        threads: ATLITE_NPROCESSES
        resources: mem_mb=ATLITE_NPROCESSES * 1000
        script: "scripts/build_cutout.py"

rule build_population_gridcell_map:
    input:
        infile="data/population/population.h5"
    output:
        population_map="data/population/population_gridcell_map.h5"
    threads: 1
    resources: mem_mb=35000
    script: "scripts/build_population_gridcell_map.py"

rule build_solar_thermal_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        profile_solar_thermal = f"data/heating/solar_thermal-{config['solar_thermal_angle']}.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_solar_thermal_profiles.py"

rule build_heat_demand_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        daily_heat_demand="data/heating/daily_heat_demand.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_heat_demand_profiles.py"

rule build_cop_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        cop="data/heating/cop.h5"
    threads: 8
    resources: mem_mb=30000
    script: "scripts/build_cop_profiles.py"

rule build_temp_profiles:
    input:
        infile="data/population/population_gridcell_map.h5"
    output:
        temp="data/heating/temp.h5"
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

if config['enable'].get('retrieve_raster', True):
    rule retrieve_build_up_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Build_up.tif"
        run: move(input[0], output[0])
    rule retrieve_Grass_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Grass.tif"
        run: move(input[0], output[0])
    rule retrieve_Bare_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Bare.tif"
        run: move(input[0], output[0])
    rule retrieve_Shrubland_raster:
        input: HTTP.remote("zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif", keep_local=True, static=True)
        output: "data/resources/Shrubland.tif"
        run: move(input[0], output[0])

rule build_renewable_potential:
    input:
        Build_up_raster="data/landuse_availability/Build_up.tif",
        Grass_raster="data/landuse_availability/Grass.tif",
        Bare_raster="data/landuse_availability/Bare.tif",
        Shrubland_raster="data/landuse_availability/Shrubland.tif",
        natura1='data/landuse_availability/WDPA_WDOECM_Mar2022_Public_CHN_shp/WDPA_WDOECM_Mar2022_Public_CHN_shp_0/WDPA_WDOECM_Mar2022_Public_CHN_shp-polygons.shp',
        natura2='data/landuse_availability/WDPA_WDOECM_Mar2022_Public_CHN_shp/WDPA_WDOECM_Mar2022_Public_CHN_shp_1/WDPA_WDOECM_Mar2022_Public_CHN_shp-polygons.shp',
        natura3='data/landuse_availability/WDPA_WDOECM_Mar2022_Public_CHN_shp/WDPA_WDOECM_Mar2022_Public_CHN_shp_2/WDPA_WDOECM_Mar2022_Public_CHN_shp-polygons.shp',
        gebco="data/landuse_availability/GEBCO_tiff/gebco_2021.tif",
        provinces_shp="data/province_shapes/CHN_adm1.shp",
        offshore_province_shapes="data/resources/regions_offshore_province.geojson",
        offshore_shapes="data/resources/regions_offshore.geojson",
        cutout= "cutouts/China-2020.nc"
    output:
        solar_profile="resources/profile_solar.nc",
        onwind_profile="resources/profile_onwind.nc",
        offwind_profile="resources/profile_offwind.nc"
    log: "logs/build_renewable_potential.log"
    threads: ATLITE_NPROCESSES
    resources: mem_mb=ATLITE_NPROCESSES * 5000
    script: "scripts/build_renewable_potential.py"

rule prepare_networks:
    input:
        population_name="data/population/population.h5",
        solar_thermal_name="data/heating/solar_thermal-{angle}.h5".format(angle=config['solar_thermal_angle']),
    	heat_demand_name="data/heating/daily_heat_demand.h5",
    	cop_name="data/heating/cop.h5",
        energy_totals_name="data/energy_totals2020.h5",
        co2_totals_name="data/co2_totals.h5",
        temp="data/heating/temp.h5",
        **{f"profile_{tech}": f"resources/profile_{tech}.nc"
           for tech in config['renewable']}
    output:
        network_name=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}.nc',
    threads: 1
    resources: mem_mb=10000
    script: "scripts/prepare_network.py"

rule solve_networks:
    input:
        network_name=config['results_dir'] + 'version-' + str(config['version']) + '/prenetworks/prenetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}.nc',
    output:
        network_name=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}.nc'
    log:
        solver=normpath("logs/solve_operations_network/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}.log")
    threads: 4
    resources: mem_mb=35000
    script: "scripts/solve_network.py"

rule plot_network:
    input:
        network=config['results_dir'] + 'version-' + str(config['version']) + '/postnetworks/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}.nc',
        tech_costs=COSTS
    output:
        only_map=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}.{ext}',
        ext=config['results_dir'] + 'version-' + str(config['version']) + '/plots/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}_ext.{ext}'
    log: "logs/plot_network/postnetwork-{flexibility}-{line_limits}-{co2_reduction}-{CHP_emission_accounting}-{opts}_{ext}.log"
    script: "scripts/plot_network.py"