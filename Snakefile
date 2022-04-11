# SPDX-FileCopyrightText: : 2012 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

from os.path import normpath, exists
from shutil import copyfile, move

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

if not exists("config.yaml"):
    copyfile("config.default.yaml", "config.yaml")

configfile: "config.yaml"
  
COSTS="data/costs.csv"
ATLITE_NPROCESSES = config['atlite'].get('nprocesses', 4)

wildcard_constraints:
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*"
    
