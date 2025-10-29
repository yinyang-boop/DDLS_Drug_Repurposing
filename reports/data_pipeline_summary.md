Here’s the English translation of your text:

---

# Data Pipeline Execution Report

Generated on: 2025-10-26 18:51:56

## Data Source Summary

* Successfully integrated data sources: **ChEMBL, PubChem**
* Final dataset size: **10,297 rows, 36 columns**

## Data Quality Metrics

* Valid compound count: **10,297**
* Activity data records: **10,297**
* Molecular descriptor completeness: **100.00%**

## Molecular Property Statistics

| Statistic | mol_weight    | logp          | hbd           | hba           | tpsa          |
| --------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| count     | 10,297.000000 | 10,297.000000 | 10,297.000000 | 10,297.000000 | 10,297.000000 |
| mean      | 480.606721    | 4.561989      | 2.202875      | 7.047198      | 95.978948     |
| std       | 121.720770    | 1.449277      | 1.187346      | 2.273566      | 30.607719     |
| min       | 110.112000    | -5.994500     | 0.000000      | 0.000000      | 0.000000      |
| 25%       | 400.334000    | 3.646920      | 1.000000      | 6.000000      | 76.720000     |
| 50%       | 480.532000    | 4.521000      | 2.000000      | 7.000000      | 96.260000     |
| 75%       | 554.655000    | 5.466270      | 3.000000      | 9.000000      | 111.630000    |
| max       | 1425.805000   | 13.028900     | 17.000000     | 21.000000     | 530.870000    |

## Recommended Next Steps

1. Check whether the data distribution is suitable for machine learning modeling.
2. Validate correlations among key molecular descriptors.
3. Consider further feature engineering and feature selection.

---
