# Independent metric reproduction

The audit independently loaded 920 reconstructed UCI rows and 3680 saved staged predictions. It verified stage/centre/source-row key uniqueness, centre labels, target values, all requested classification metrics, calibration intercept and slope, and the stored metric values. Bootstrap interval comparisons allow a small tolerance because the earlier implementation used an independent bootstrap stream; point estimates are compared at numerical precision.

The prediction table cannot by itself prove the historical fit provenance. The source implementation was inspected separately: preprocessing and parameter selection are inside development-centre fits, while the held-out centre supplies only predictions for each outer fold.
