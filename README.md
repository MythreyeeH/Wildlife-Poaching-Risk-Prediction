# Wildlife Poaching Risk Analysis

**An Unsupervised Machine Learning Framework for Species-Wise Anomaly Detection**


## Overview

This project implements a **data science pipeline** to identify **high-risk poaching origins** by analyzing global wildlife trade behavior.

Rather than evaluating individual trade records, the system analyzes **aggregated behavioral profiles** to surface **systematic anomalies** that may indicate illegal or high-risk trade patterns.

The core modeling unit is defined as:

> **(Taxon, Exporter Country)**

This aggregation reduces transactional noise and captures long-term behavioral signals.

---

## Modeling Methodology

- **Unit of Analysis:** `(Taxon, Exporter)`
- **Rationale:**  
  Poaching risk arises from **consistent trade behavior patterns**, not isolated trade events.  
  Aggregating transactions creates a stable **behavioral signature** for each species–exporter pair, enabling detection of systemic irregularities rather than one-off reporting errors.

This design improves robustness against data noise and reporting inconsistencies.

---

## Feature Engineering

After grouping the dataset by **Taxon** and **Exporter**, a behavioral profile is constructed using the following features:

| Feature            | Description |
|--------------------|-------------|
| `App.`             | Mean CITES appendix severity (protection level) |
| `import_qty_log`   | Log-transformed total imported quantity |
| `export_qty_log`   | Log-transformed total exported quantity |
| `num_trade_events` | Frequency of trade occurrences |
| `is_live`          | Proportion of live-animal trade |
| `purpose`          | Mean risk score based on trade purpose |
| `source `          | Mean risk score based on trade source |

**Note:**  
Log transformations are applied to quantity-based features to mitigate skew and handle extreme variance in trade volumes.

---

## Machine Learning Pipeline

### Data Scaling

- **Scaler:** `RobustScaler`
- **Why:**  
  Wildlife trade data is inherently noisy and contains extreme outliers.  
  `RobustScaler` uses the **interquartile range (IQR)**, preventing unusually large trade volumes from dominating the feature space.

---

### Model Selection

- **Model:** `Isolation Forest`
- **Justification:**  
  Illegal or poaching-related trade is assumed to be **rare and anomalous**.  
  Isolation Forests isolate anomalies by recursively partitioning the feature space, making them well-suited for **high-dimensional, unlabeled data**.

---

### Key Hyperparameters

- `n_estimators = 300`  
  Ensures a stable ensemble and reliable anomaly score convergence.

- `contamination = 0.05 – 0.1`  
  Specifies the expected proportion of anomalous behavior in the dataset.

---

## Risk Scoring Mechanism

To convert raw anomaly outputs into an interpretable **Risk Index**, the following transformations are applied:

### Inversion
Raw Isolation Forest scores are inverted so that **higher values indicate higher risk**.

### Normalization
Scores are scaled to a **0.0 – 1.0 range** using Min–Max scaling.

### Interpretation

- **0.0** → Typical trade behavior (Low Risk)
- **1.0** → Highly anomalous trade behavior (High Risk / Potential Poaching Origin)

---

## Validation Strategy

Because illegal poaching activity lacks labeled ground truth, validation is performed using **domain-driven evaluation** rather than standard supervised metrics.

### CITES Appendix Alignment

Species listed under **stricter CITES appendices** (e.g., Appendix I) are expected to receive **higher risk scores**, reflecting increased regulatory protection and trafficking pressure.

### Geographic Consistency

High-risk exporter countries identified by the model are compared against **historically documented poaching and trafficking hotspots** for the corresponding taxa.

Alignment with known conservation reports provides **qualitative validation** of model behavior.
