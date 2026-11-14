# NHANES Extraction Protocol

The v2 pipeline pins the 2017-2018 cycle, downloads component XPT files and HTML documentation, merges only on SEQN, checks duplicate SEQN values, and preserves SDMVPSU, SDMVSTRA, WTINT2YR, and WTMEC2YR when present. No synthetic cardiovascular target is created and survey weights must be used for population-level inference.
