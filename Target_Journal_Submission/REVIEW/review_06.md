# Review round 6 — computational cost, memory, scalability, deployment

| # | Severity | Issue | Action | Status |
|---|---|---|---|---|
| 1 | MAJOR | An early draft called the correction "essentially free". Measured: LOO-EM costs a median 2.6× (mean ≈ 0.3 s) the eigendecomposition; only one-round LOO relabelling is nearly free. | Wording changed everywhere: "cheap", "a small multiple of the eigendecomposition"; RQ3 keeps the question form and Sect. 6.7 answers it with the measured ratio. | Fixed |
| 2 | MAJOR | O(n³) time and O(n²) memory: three n × n arrays (eigenvectors, their squares, centred features Gram). ElectricDevices (8,926 series) was not completed on the 15 GB machine within the time available. | Scalability experiment restricted to FordA, FordB and Crop (3,601–7,200 series) and reported as such; the limit (~10⁴ series on a workstation) is in Limitations. ElectricDevices is **not** reported. | Disclosed |
| 3 | MODERATE | Is LOO-EM worth 0.3 s for ≤ 2 points? | Discussion paragraph "Is the improvement worth its cost?" answers: the grid is free; LOO-EM is optional and regime-specific; cheaper than 5-fold confident learning by about the ratio reported in Sect. 6.7. | Addressed |
| 4 | MODERATE | Feature extraction dominates for small n (median 0.5 s vs 0.03 s eigendecomposition); for large n the eigendecomposition dominates. | Both reported; timing table for the large datasets in the supplement. | Fixed |
| 5 | MODERATE | Energy, FLOPs, model size not reported. | Not measured (no power instrumentation); the deployed model is the same linear classifier in all ridge variants, so inference cost and model size are identical by construction — stated in Sect. 6.7. | Disclosed |
| 6 | MINOR | Timings obtained on a shared 4-core VM with other jobs running. | Stated; ratios to the eigendecomposition measured in the same process are the robust quantity. | Disclosed |
| 7 | MINOR | The feature cache needs ~6 GB per seed. | README; cache limited to seeds 0–2 after a disk-full incident during the study. | Fixed |
