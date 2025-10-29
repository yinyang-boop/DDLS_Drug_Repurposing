# Executive Summary

This project delivers a **reproducible, engineered pipeline for drug–target affinity (DTA) prediction**, with a focus on **peptide drug repurposing for rare disease targets** such as the Cystic Fibrosis Transmembrane Conductance Regulator (CFTR). The platform integrates cheminformatics, sequence analysis, and deep learning into a modular, auditable framework called the **MCP Toolset**.

**Motivation and Goal**  
Drug repurposing is a cost‑effective strategy for accelerating therapeutic discovery, but it requires robust pipelines capable of handling heterogeneous data and ensuring reproducibility. Our goal was to establish a reliable baseline using the **DeepDTA model** (1D CNN) on a benchmark oncology target (EGFR), and then extend the platform toward trans‑target prediction for CFTR.

**Pipeline Design**  
- **MCP Toolset (Python package `mcp-pipeline`)**:  
  - *MolecularAnalyzer*: computes descriptors, fingerprints, and similarity scores.  
  - *SequenceAnalyzer*: quantifies peptide/protein similarity.  
  - *DeepDTAProcessor*: encodes SMILES and sequences, cleans datasets, and ensures reproducible splits.  
- **Automated Testing**: Comprehensive test scripts generate Markdown reports and visualizations (bar charts, heatmaps, histograms), ensuring transparency and auditability.  
- **Workflow Orchestration**: Prefect 2.x enables reproducible execution across environments.

**Key Results**  
- **Molecular Analysis**: Semaglutide and Tirzepatide share high structural similarity (Tanimoto = 0.715), with expected differences in molecular weight and polarity.  
- **Sequence Analysis**: Identical GLP‑1 backbones yield perfect similarity scores; partially similar sequences are correctly quantified.  
- **Dataset Processing**: A curated ChEMBL/PubChem dataset (13,286 entries) was cleaned and reproducibly split into training (10,628), validation (1,329), and test (1,329).  
- **DeepDTA Model Performance**: On EGFR, the model achieved **Concordance Index (Ci) = 0.85**, **MSE = 0.22**, and estimated **ROC‑AUC ≈ 0.90**, confirming strong predictive utility.  
- **Reproducibility**: Automated reports and figures document every step, ensuring FAIR compliance.

**Implications**  
The validated MCP Toolset demonstrates that **engineering integrity and reproducibility** can deliver high‑quality scientific outcomes under resource constraints. The DeepDTA baseline provides a strong foundation for **trans‑target prediction** on CFTR and positions the platform for **next‑generation GNN integration** with deferred datasets (GDSC, BindingDB, DrugBank). This roadmap aligns with the broader vision of advancing toward **precision polypharmacology**.

---

