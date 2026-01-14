# 📚 Notes on RecSys Models

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.8%2B-green) ![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)

> **My notes on models in recommendation systems. All models are implemented in PyTorch, with simple usage demos and brief introduction markdown notes.**

Welcome to **Notes-on-RecSys-Models**! This repository is a collection of my learning notes and PyTorch implementations of classic and state-of-the-art recommendation system algorithms.

The goal of this project is to provide:
1.  **Clean Code:** Readable PyTorch implementations focused on the model structure.
2.  **Theory:** Brief markdown notes explaining the core idea behind each model.
3.  **Practice:** Simple demos to show how to input data and get predictions.

## 📂 Directory Structure

Here is the organization of the project:

```text
Notes-on-RecSys-Models/
├── assets/                 # Images and diagrams for documentation
├── Classic Models/         # Model implementations organized by stage
│   ├── Recall/             # Retrieval models
│   ├── Ranking/            # Fine-ranking models
│   └── Re-ranking/         # Re-ranking strategies
├── Data/                   # Raw data storage (e.g., Amazon Review 2018)
├── Datasets/               # PyTorch Dataset wrappers and preprocessing logic
│   └── BaseDatasets.py     # Base classes for datasets
├── Utils/                  # Common utility functions (I/O, metrics)
└── README.md
```

## 🚀 Implemented Models

Models are organized by the stage they are typically used in a recommendation pipeline: **Recall** (Retrieval), **Ranking** (Fine-sorting), and **Re-ranking**.

### 1. Recall
> *Goal: Retrieve a small set of relevant items from a massive pool of candidates efficiently.*

| Model | Paper / Source | Code | Note | Status |
| :--- | :--- | :---: | :---: | :---: |
| **YoutubeDNN** | [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf) | [Link](Classic%20Models/Recall/YouTubeDNN/model.py) | [📝](Classic%20Models/Recall/YouTubeDNN/README.md ) | 📅 |
| **MIND** | [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030) | [Link](Classic%20Models/Recall/MIND/model.py) | [📝](Classic%20Models/Recall/MIND/README.md ) | ✅ |

### 2. Ranking
> *Goal: Accurately predict user preference (CTR/CVR) for the retrieved items to sort them.*

| Model | Paper / Source | Code | Note | Status |
| :--- | :--- | :---: | :---: | :---: |

### 3. Re-ranking
> *Goal: Optimize the final list for diversity, novelty, or business constraints.*

| Model | Source / Concept | Code | Note | Status |
| :--- | :--- | :---: | :---: | :---: |

(Legend: ✅ Implemented, 🚧 In Progress, 📅 Planned)