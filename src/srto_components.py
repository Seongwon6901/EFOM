

# ------------------------------------------------------------------------
# 1. Define Component Index and Molecular Weights (from your code)
# ------------------------------------------------------------------------
component_index = {
    'Benzene' : 25,
    'Toluene' : 26,

    # ─── C0/C1 Gases ───────────────────────────
    'H2': 11,          # Hydrogen
    'CH4': 12,         # Methane

    # ─── C2 ────────────────────────────────────
    'C2H2': 13,        # Acetylene
    'C2H4': 14,        # Ethylene
    'Ethylene': 14,
    'C2H6': 15,        # Ethane
    'Ethane': 15,

    # ─── C3 ────────────────────────────────────
    'C3H4': 16,        # Allene/Propyne (use as proxy)
    'C3H6': 17,        # Propylene
    'Propylene': 17,
    'C3H8': 18,        # Propane
    'Propane': 18,

    # ─── C4 ────────────────────────────────────
    'NBUTA': 19,       # n-Butane
    'n-Butane': 19,
    'IBUTA': 20,       # i-Butane
    'i-Butane': 20,
    'Butadiene': 24,   # 1,3-Butadiene
    'C4 n-Paraffin': 24,    # n-Butane
    'C4 i-Paraffin': 25,    # i-Butane
    # C4 olefins are often not explicit; 69 is C8-olefins lump, but use if that's the closest proxy.
    'C4 Olefin': 69,

    # ─── C5 ────────────────────────────────────
    'C5 n-Paraffin': 31,    # n-Pentane (was incorrectly 16 before)
    'C5 i-Paraffin': 32,    # i-Pentane
    'C5 Olefin': 46,        # 1-Pentene (can add isoprene as 52 if needed)
    'C5 Naphthene': 28,     # Cyclopentane

    # ─── C6 ────────────────────────────────────
    'C6 n-Paraffin': 33,    # n-Hexane
    'C6 i-Paraffin': 55,    # i-Hexane
    'C6 Olefin': 53,        # 1-Hexene
    'C6 Naphthene': 29,     # Methyl cyclopentane
    'C6 Aromatic': 25,      # Benzene

    # ─── C7 ────────────────────────────────────
    'C7 n-Paraffin': 34,           # n-Heptane
    'C7 i-Paraffin': 60,           # Iso-heptane
    'C7 Olefin': 58,               # 1-Heptene
    'C7 Aromatic (toluene)': 26,   # Toluene
    'C7 Aromatic': 26,             # Toluene (xylenes actually C8)
    'C7 Naphthene': 42,            # C7 Naphthene

    # ─── C8 ────────────────────────────────────
    'C8 n-Paraffin': 35,    # n-Octane
    'C8 i-Paraffin': 65,    # Iso-octane
    'C8 Olefin': 63,        # C8 Olefin
    'C8 Naphthene': 43,     # C8 Naphthene
    # Aromatics - combine xylene/ethylbenzene/styrene if needed
    'C8 Aromatic': 27,      # Xylenes 27
    'Ethyl benzene': 28,    # Ethylbenzene
    'Styrene': 29,          # Styrene (if present)
    'C8 Cycloalkane': 30,   # Cyclohexane (proxy for cyclooctane if needed)

    # ─── C9 ────────────────────────────────────
    'C9 n-Paraffin': 36,    # n-Nonane
    'C9 i-Paraffin': 67,    # i-Nonane
    'C9 Olefin': 70,        # 1-Nonene
    'C9 Naphthene': 44,     # C9 Naphthene
    'C9 Aromatic': 40,      # C9 Aromatic (best fit)

    # ─── C10 ───────────────────────────────────
    'C10 n-Paraffin': 41,   # n-Decane
    'C10 i-Paraffin': 71,   # i-Decane
    'C10 Olefin': 73,       # 1-Decene
    'C10 Naphthene': 45,    # C10 Naphthene
    'C10 Aromatic': 41,     # C10 Aromatic (best fit)

    # ─── C11+ ──────────────────────────────────
    'C11+ n-Paraffin': 78,
    'C11+ i-Paraffin': 85,
    'C11+ Naphthene': 92,
    'C11+ Aromatic': 109,

    # ─── Other Special Cases (Lab Report) ──────
    'Methyl Cyclopentane': 29,
    'Methyl Cyclohexane': 30,
    'Cyclohexane': 30,      # same as Methyl Cyclohexane if cyclohexane not explicit

    # ─── Other "lump" or heavier proxies ───────
    # 'C12+': (add if needed),
}


# ------------------------------------------------------------------------
# 2. Component → SPYIN index map (keep your full dictionary here)
# ------------------------------------------------------------------------

MW = {
    'Benzene': 6*12.011 + 6*1.008,
    'Toluene': 7*12.011 + 8*1.008,

    # Gases
    'H2':                     2*1.008,
    'CH4':        12.011 + 4*1.008,
    'C2H2': 2*12.011 + 2*1.008,
    'C2H4': 2*12.011 + 4*1.008,
    'Ethylene':    2*12.011 + 4*1.008,
    'C2H6': 2*12.011 + 6*1.008,
    'Ethane':      2*12.011 + 6*1.008,
    'C3H4': 3*12.011 + 4*1.008,
    'C3H6': 3*12.011 + 6*1.008,
    'Propylene':   3*12.011 + 6*1.008,
    'C3H8': 3*12.011 + 8*1.008,
    'Propane':     3*12.011 + 8*1.008,
    'NBUTA':4*12.011 +10*1.008,
    'n-Butane':    4*12.011 +10*1.008,
    'IBUTA':4*12.011 +10*1.008,
    'i-Butane':    4*12.011 +10*1.008,

    # Light paraffins/olefins/naphthenes/aromatics
    'Methyl Cyclopentane':    6*12.011 +12*1.008,   # C6H12
    'Methyl Cyclohexane':     7*12.011 +14*1.008,   # C7H14
    'Ethyl benzene':          8*12.011 +10*1.008,   # C8H10
    'Cyclohexane':            6*12.011 +12*1.008,   # C6H12
    'C4 n-Paraffin':          4*12.011 +10*1.008,   # C4H10
    'C4 i-Paraffin':          4*12.011 +10*1.008,
    'C4 Olefin':              4*12.011 + 8*1.008,   # C4H8
    'C5 n-Paraffin':          5*12.011 +12*1.008,   # C5H12
    'C5 i-Paraffin':          5*12.011 +12*1.008,
    'C5 Olefin':              5*12.011 +10*1.008,   # C5H10
    'C5 Naphthene':           5*12.011 +10*1.008,   # cyclopentane
    'C6 n-Paraffin':          6*12.011 +14*1.008,   # C6H14
    'C6 i-Paraffin':          6*12.011 +14*1.008,
    'C6 Olefin':              6*12.011 +12*1.008,   # C6H12
    'C6 Naphthene':           6*12.011 +12*1.008,   # methyl cyclopentane proxy
    'C6 Aromatic':            6*12.011 + 6*1.008,   # benzene (C6H6)
    'C7- Aromatic':           7*12.011 + 8*1.008,   # toluene (C7H8)
    'C7 n-Paraffin':          7*12.011 +16*1.008,   # C7H16
    'C7 i-Paraffin':          7*12.011 +16*1.008,
    'C7 Olefin':              7*12.011 +14*1.008,   # C7H14
    'C7 Aromatic':            7*12.011 + 8*1.008,   # toluene proxy for xylenes
    'C7 Naphthene':           7*12.011 +14*1.008,   # cycloheptane
    'C7 Naphthene(5-Ring)':   7*12.011 +14*1.008,

    'C8- Non-Aromatic':       8*12.011 +18*1.008,   # C8H18
    'C8 n-Paraffin':          8*12.011 +18*1.008,
    'C8 i-Paraffin':          8*12.011 +18*1.008,
    'C8 Olefin':              8*12.011 +16*1.008,   # C8H16
    'C8 Naphthene':           8*12.011 +16*1.008,
    'C8 Aromatic':            8*12.011 +10*1.008,   # xylene (C8H10)
    'C8 Naphthene(5-Ring)':   8*12.011 +16*1.008,
    'C8 Naphthene(6-Ring)':   8*12.011 +16*1.008,
    'C8+ Aromatic':           9*12.011 +12*1.008,   # C9H12 proxy

    'C9 n-Paraffin':          9*12.011 +20*1.008,   # C9H20
    'C9 i-Paraffin':          9*12.011 +20*1.008,
    'C9 Olefin':              9*12.011 +18*1.008,   # C9H18
    'C9 Naphthene':           9*12.011 +18*1.008,
    'C9 Aromatic':            9*12.011 +10*1.008,   # C9H10
    'C9 Naphthene(5-Ring)':   9*12.011 +18*1.008,
    'C9 Naphthene(6-Ring)':   9*12.011 +18*1.008,
    'C9+ Aromatic':          10*12.011 +12*1.008,   # C10H12 proxy

    'C9+ Non-Aromatic':      10*12.011 +22*1.008,   # decane (C10H22)
    'C10 n-Paraffin':        10*12.011 +22*1.008,
    'C10 i-Paraffin':        10*12.011 +22*1.008,
    'C10 Olefin':            10*12.011 +20*1.008,   # C10H20
    'C10 Naphthene':         10*12.011 +20*1.008,
    'C10 Aromatic':          10*12.011 + 8*1.008,   # naphthalene (C10H8)
    'C10 Naphthene(6-Ring)': 10*12.011 +18*1.008,   # C10H18
    'n-nonane':               9*12.011 +20*1.008,
    'C10+ Aromatic':         12*12.011 +18*1.008,   # C12H18 proxy

    'C11+ n-Paraffin':       12*12.011 +26*1.008,   # C12H26
    'C11+ i-Paraffin':       12*12.011 +26*1.008,
    'C11+ Naphthene':        12*12.011 +24*1.008,   # C12H24
    'C11+ Aromatic':         12*12.011 +18*1.008,   # C12H18
}