# Adversarial Detection with GA-Ensemble

---

## Overview

This repository contains code for adversarial example detection on image classification systems. The pipeline combines a **Hybrid Stacking Ensemble** of unsupervised detectors and supervised classifiers, with a **Genetic Algorithm (GA)** to select the optimal detector subset for each attack scenario.

---

## Repository Structure
```text
source.ipynb   # Training pipeline with GA-optimized stacking
demo.ipynb     # Inference notebook with interactive interface
```

---

## Datasets & Pretrained Weights

Download and upload to Kaggle as datasets before running notebooks.

- **Attacked .pth files**: [Kaggle](https://www.kaggle.com/datasets/seapanther/attacked-pth-files)
- **CIFAR-10, CIFAR-100, SVHN (batched)**: [Kaggle](https://www.kaggle.com/datasets/seapanther/downloaded-datasets)
- **Pretrained ResNet weights**: Originally from [Deep Mahalanobis Detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector) (original links no longer available). Download from our Kaggle mirror: [Kaggle](https://www.kaggle.com/datasets/sealeopard/resnet-pth)

---

## Running the Notebooks

All notebooks are designed to run on **Kaggle** (free GPU).

1. Upload datasets and pretrained weights to Kaggle as datasets.
2. Open the desired notebook on Kaggle.
3. Update dataset paths if necessary.
4. Run all cells.

### source.ipynb — Training

Configure the following variables before running:
```python
ds_name = "cifar10"       # Options: "cifar10", "svhn"
adv_type = "FGSM"         # Options: "FGSM", "BIM", "DeepFool", "CWL2"
adv_transfer_type = "DeepFool"  # Options: "FGSM", "BIM", "CWL2"
```

- `run_enad_binary_stack_transfer` — runs GA-stacking for transfer attack scenarios
- `genetic_algorithm` — optimizes stacking configuration for the selected scenario

### demo.ipynb — Inference

Run all cells. An interactive interface will appear with the following options:
```python
ds_name = st.selectbox("Select Dataset", ["cifar10", "svhn"])
adv_type = st.selectbox("Select Adversarial Attack Type", ["FGSM", "BIM", "DeepFool", "CWL2"])
algorithm = st.selectbox("Select Algorithm", ["enad", "enad_ga"])
```

---

## Results

Achieved up to **76% accuracy improvement** against transfer attacks compared to baseline stacking methods across 4 adversarial attack benchmarks.

---

## Acknowledgements

- [ENAD Experiments](https://github.com/BIMIB-DISCo/ENAD-experiments)
- [Deep Mahalanobis Detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)
