# Research and Standards References

ProcureLens is implementation-driven, but its analytical boundaries are informed by public procurement guidance, federal source documentation, statistical anomaly-detection research, and reproducible machine-learning practice.

These references are **design inputs**, not claims that ProcureLens exactly reproduces every cited method or that any cited red flag proves wrongdoing.

## Federal procurement and source semantics

- **USAspending API documentation** — source access, award/search endpoints, and federal spending data semantics: https://api.usaspending.gov/docs/
- **Federal Acquisition Regulation (FAR), Part 6 — Competition Requirements** — distinguishes competition procedures from the results of competition and documents circumstances permitting other than full and open competition: https://www.acquisition.gov/far/part-6
- **Acquisition.gov / FPDS reporting guidance** — used when interpreting federal procurement reporting fields and lifecycle semantics: https://www.acquisition.gov/

## Procurement integrity and competition context

- **OECD Guidelines for Fighting Bid Rigging in Public Procurement (2025 Update)** — market structure, bidding-pattern red flags, detection guidance, and the need for contextual interpretation. DOI: https://doi.org/10.1787/cbe05a56-en
- **OECD Anti-Corruption and Integrity Outlook 2026 — Integrity in Public Procurement** — discusses indicators such as single bidding, non-competitive procedures, repeated awards, and contract modifications while warning against oversimplified conclusions. DOI: https://doi.org/10.1787/16708b78-en

## Anomaly detection

- Liu, F. T., Ting, K. M., and Zhou, Z.-H. **“Isolation Forest.”** IEEE ICDM, 2008. Public author copy: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
- Liu, F. T., Ting, K. M., and Zhou, Z.-H. **“Isolation-based Anomaly Detection.”** ACM TKDD, 2012. Public author copy: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf
- Hariri, S., Carrasco Kind, M., and Brunner, R. J. **“Extended Isolation Forest.”** 2019. https://arxiv.org/abs/1811.02141 — relevant to known axis-aligned Isolation Forest scoring artifacts; ProcureLens currently keeps standard Isolation Forest as one ensemble member rather than treating it as ground truth.
- Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., and Chen, G. H. **“ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions.”** 2022. https://arxiv.org/abs/2201.00382 — informs the idea of empirical per-feature tail evidence. ProcureLens uses its own frozen-training empirical-tail implementation and does **not** claim exact ECOD compatibility.

## Explainability and reproducible ML practice

- Phillips, P. J. et al. **“Four Principles of Explainable Artificial Intelligence.”** NISTIR 8312, 2021. https://doi.org/10.6028/NIST.IR.8312 — supports the requirement that explanations remain meaningful and faithful to what the system actually computes.
- **scikit-learn IsolationForest documentation** — estimator score/configuration semantics used by the adapter: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- **scikit-learn common pitfalls and recommended practices** — train/test separation and preprocessing leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html

## Interpretation boundary

ProcureLens uses these materials to justify distinctions such as:

- competition process vs. number of offers received;
- red flag vs. proof of misconduct;
- base/new awards vs. later modifications;
- reference/training populations vs. scoring populations;
- raw detector score vs. calibrated review priority;
- evidence explanation vs. unsupported causal feature attribution.

Real-world accuracy still requires validation on deliberately selected live data and, where available, external review or ground truth.
