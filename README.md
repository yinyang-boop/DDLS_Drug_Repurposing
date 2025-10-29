# 🧬 Drug Repurposing Research Platform

A reproducible and modular computational pipeline (MCP) for **Drug-Target Affinity (DTA) prediction** and **peptide drug repurposing**.  
This project implements a robust workflow around the **DeepDTA** model, cheminformatics analysis with **RDKit**, and workflow orchestration with **Prefect**.

---

## 🚀 Features

- **Data Acquisition**: Automated retrieval and parsing from ChEMBL, PubChem, and other sources.
- **Cheminformatics Tools**: Molecular descriptors, fingerprints, and similarity scoring via RDKit.
- **Sequence Analysis**: Protein/peptide sequence similarity (identity, Levenshtein).
- **Deep Learning**: DeepDTA (1D CNN) for drug–target affinity prediction.
- **Workflow Orchestration**: Prefect 2.x pipelines for reproducible execution.
- **Visualization**: Descriptor plots, similarity heatmaps, dataset statistics.

---

## 📦 Dependencies

### Data Acquisition Clients
- `chembl-webresource-client`
- `pubchempy`
- `requests`

### Core Data Science & ML
- `pandas`
- `numpy`
- `scikit-learn`

### Cheminformatics & Visualization
- `rdkit-pypi`
- `matplotlib`
- `seaborn`
- `plotly`

### Deep Learning (DeepDTA Model)
- `torch`
- `torch-geometric`

### Workflow Orchestration
- `prefect>=2.0`

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yinyang-boop/DDLS_Drug_Repurposing.git
cd DDLS_Drug_Repurposing
pip install -e .
```

Or install directly with pip:

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

### 1. Run molecular analysis
```bash
python test_mcp_tools.py
```
Generates descriptors, similarity scores, sequence metrics, and a Markdown report (`REPORT.md`) with visualizations.

### 2. Use as a package
```python
from mcp_pipeline import MolecularAnalyzer, SequenceAnalyzer, DeepDTAProcessor

desc = MolecularAnalyzer.calculate_descriptors("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
print(desc)
```

---

## 📊 Example Outputs

- Molecular descriptor comparison bar charts
- Tanimoto similarity heatmaps
- Sequence similarity bar plots
- DeepDTA dataset size distributions

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📜 License

MIT License © 2025 Yin Yang

---

## 🔗 References

- Öztürk et al. (2018) *DeepDTA: Deep Drug–Target Binding Affinity Prediction*.  
- RDKit: [https://www.rdkit.org](https://www.rdkit.org)  
- Prefect: [https://www.prefect.io](https://www.prefect.io)  
