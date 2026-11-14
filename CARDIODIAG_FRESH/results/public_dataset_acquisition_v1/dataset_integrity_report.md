# Dataset integrity report

- UCI combined rows: **920**.
- UCI site counts: {'cleveland': 303, 'hungary': 294, 'switzerland': 123, 'va_long_beach': 200}.
- UCI raw target counts: {0: 411, 1: 265, 2: 109, 3: 107, 4: 28}.
- UCI binary target counts (`num > 0`): {np.int64(0): 411, np.int64(1): 509}.
- UCI exact duplicate occurrences: **2**, in two within-site groups; no cross-site exact duplicate group was observed.
- AngioCAD feature rows/unique IDs: **377/377**.
- AngioCAD label rows/unique IDs: **413/413**.
- AngioCAD feature-label ID intersection: **377**; label-only IDs: **36**.
- AngioCAD derived lower-bound >=51% artery stenosis labels: {np.int64(0): 89, np.int64(1): 324}.

The AngioCAD patient-level label is diagnostic-only and derived for audit; it must not be used as a replacement for a prespecified clinical endpoint without reading the variable dictionary and paper.
