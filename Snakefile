# SPDX-FileCopyrightText: : 2012 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# Build capacities in China

rule build_p_nom:
    output:
        coal_capacity = "data/p_nom/coal_p_nom.h5"
        CHP_capacity="data/p_nom/CHP_p_nom.h5"
        OCGT_capacity="data/p_nom/OCGT_p_nom.h5"
    threads:1
    resources: mem_mb=500
    script: "scripts/build_p_nom.py"
