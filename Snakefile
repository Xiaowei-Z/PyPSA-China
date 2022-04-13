# SPDX-FileCopyrightText: : 2012 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# Build capacities in China

rule build_p_nom:
    input:
    output:
    threads:1
    resources: mem_mb=500
    script: "scripts/build_p_nom.py"
