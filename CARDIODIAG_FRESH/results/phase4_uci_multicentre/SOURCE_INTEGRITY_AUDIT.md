# Source integrity audit

The official UCI archive was acquired in the preceding public-data phase and its SHA-256 is preserved in `SHA256_MANIFEST.json`. The archive contains all four processed source files. Independent reconstruction yields **920 records**: {'Cleveland': 303, 'Hungary': 294, 'Switzerland': 123, 'VA Long Beach': 200}. The processed files contain 14 columns, use `?` for missing values, and have no patient identifiers.

The source documentation defines `num=0` as less than 50% diameter narrowing and `num=1–4` as greater than 50% diameter narrowing. This study uses `target_binary = (num > 0)` without changing the source endpoint.
