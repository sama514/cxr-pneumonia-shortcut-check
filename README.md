# Chest X-ray Classification: NORMAL vs PNEUMONIA
A small, honest, CPU-friendly pipeline that classifies chest X-rays as NORMAL or PNEUMONIA, and, more importantly, **checks whether the result can be trusted**.

Most public tutorials on this dataset report accuracy above 95% and stop there. This project asks a harder question: *is the model looking at the lungs, or at something else in the image?*

---

## Table of contents

1. [Headline result](#headline-result)
2. [What makes this project different](#what-makes-this-project-different)
3. [Dataset](#dataset)
4. [Method](#method)
5. [Results](#results)
6. [The shortcut finding](#the-shortcut-finding)
7. [Project structure](#project-structure)
8. [How to run](#how-to-run)
9. [Reproducibility notes](#reproducibility-notes)
10. [Limitations](#limitations)
11. [Future work](#future-work)
12. [References](#references)

---

## Headline result

| Model | AUC (95% CI) | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Main model (frozen ResNet18 + logistic regression, threshold 0.5) | **0.991** (0.986 to 0.996) | 0.963 | 0.970 | 0.943 |
| Shortcut model (image size + brightness only) | **0.942** (0.924 to 0.959) | 0.874 | 0.931 | 0.725 |

Test set: **889 images from 420 patients**, split by patient with no overlap between train, validation and test.

> **Important finding:** a model that sees only image size and overall brightness reaches an AUC of 0.942. The dataset therefore contains a strong shortcut, and the main model's AUC of 0.991 cannot be read as lung-based performance alone. Treat these numbers as an **upper bound** until they are validated on external data.

---

## What makes this project different

**1. Split by patient, not by image.**
One patient can have several X-rays. If images of the same patient land in both training and test sets, the model can recognise the person instead of the disease, and the reported numbers become inflated. This project extracts the patient ID from each file name, groups images by patient, and verifies that no patient appears on both sides of any split.

**2. A shortcut check.**
A tiny model is trained on properties that have nothing to do with lung tissue: image width, height and average brightness. If that model scores well, the dataset leaks the label through how the images were acquired, and the main model's score is suspect.

**3. The official Kaggle split is ignored on purpose.**
The official validation folder contains only 16 images, which is far too small to tune anything or to give a stable estimate. All images are pooled and re-split by patient.

**4. Light enough for a laptop.**
A frozen ResNet18 is used as a feature extractor and a logistic regression is trained on top. Everything runs on CPU. Embeddings are cached to disk, so every run after the first one is fast.

**5. Honest metrics.**
Pneumonia is about 72% of the data, so a model that always answers "pneumonia" already gets about 72% accuracy. Accuracy alone is misleading. This project reports AUC with bootstrap confidence intervals, plus sensitivity and specificity.

---

## Dataset

- **Name:** Chest X-Ray Images (Pneumonia), available on Kaggle.
- **Classes:** `NORMAL` and `PNEUMONIA`.
- **Class balance:** roughly 72% pneumonia, 28% normal.
- **Source:** pediatric chest X-rays. Images were collected in clinical routine care, so they differ in size, contrast and acquisition conditions. This is exactly what makes shortcuts possible.
- **Not included in this repository.** Download it yourself from Kaggle (a free account is required) and respect its license. Do not commit the images.

Expected layout (the exact sub-folder structure under `data/` does not matter, the loader searches recursively):

```
data/
  chest_xray/
    train/  NORMAL/  PNEUMONIA/
    test/   NORMAL/  PNEUMONIA/
    val/    NORMAL/  PNEUMONIA/
```

Folders such as `__MACOSX` that appear after unzipping are harmless and are ignored.

---

## Method

### 1. Read the data and recover patient IDs
All image paths are collected and the patient identifier is parsed from the file name. Patients are the unit of splitting, not images.

### 2. Split by patient
Patients are divided into train, validation and test groups. A check confirms that the intersection of patient sets between any two groups is empty. The test set ends up with 889 images from 420 patients.

### 3. Extract embeddings with a frozen ResNet18
Each image is passed through a pretrained ResNet18 with its classification head removed. The resulting vector (the embedding) summarises the image. The network is **not** fine-tuned. Embeddings are stored in `cache/emb_resnet18.npz`, so this slow step happens only once.

### 4. Train a logistic regression
A logistic regression is fit on the embeddings of the training patients. Hyperparameters are chosen on the validation patients only. The test set is touched once, at the end.

### 5. Evaluate
On the test set the project computes:
- **AUC**, with a 95% confidence interval from bootstrap resampling.
- **Accuracy**, **sensitivity** (recall on pneumonia) and **specificity** (recall on normal) at a 0.5 threshold.

### 6. Shortcut check
A second logistic regression is trained using only the image width, height and mean brightness, using the same split. Its AUC is compared with the main model.

### 7. Report
Results are written to `reports/metrics.json` (machine readable) and `reports/metrics.md` (human readable).

---

## Results

### Main model

| Metric | Value |
|---|---|
| AUC | 0.991 (95% CI 0.986 to 0.996) |
| Accuracy | 0.963 |
| Sensitivity | 0.970 |
| Specificity | 0.943 |

How to read this:
- **Sensitivity 0.970** means about 97 out of 100 pneumonia cases are caught.
- **Specificity 0.943** means about 94 out of 100 healthy cases are correctly cleared.
- The AUC confidence interval is narrow, so the estimate is stable on this test set. It says nothing about whether the model would hold up on images from another hospital.

### Shortcut model (size and brightness only)

| Metric | Value |
|---|---|
| AUC | 0.942 (95% CI 0.924 to 0.959) |
| Accuracy | 0.874 |
| Sensitivity | 0.931 |
| Specificity | 0.725 |

This model never sees lung content. It is only given how big the image is and how bright it is on average.

### Side by side

| Model | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Main (ResNet18 + LR) | 0.991 | 0.963 | 0.970 | 0.943 |
| Shortcut (size + brightness) | 0.942 | 0.874 | 0.931 | 0.725 |
| Gap | 0.049 | 0.089 | 0.039 | 0.218 |

The gap in AUC is small. The largest difference is in specificity: the real model is much better at recognising healthy lungs, which is a sign that it does extract some information beyond size and brightness. But most of the headline performance can already be reached without looking at the lungs.

---

## The shortcut finding

The decision rule used in this project:

- **AUC near 0.5:** no obvious shortcut from these simple properties.
- **AUC clearly above 0.5:** the dataset leaks the label through acquisition properties, so part of the main model's score may not come from the lungs.
- **Compare with the main model AUC.** If they are close, be suspicious of the main result.

**What happened here:** the shortcut model reached 0.942, far above 0.5 and fairly close to the main model's 0.991. The dataset has a strong shortcut.

**Likely reason (a hypothesis, not something proven here):** pneumonia and normal images were probably collected, exported or processed differently, so image dimensions and exposure differ systematically between classes. A model can pick this up without knowing anything about lungs.

**Why this matters:** a model like this can look excellent on a held-out test set from the same source and then fail on a new hospital, a different scanner or a different export pipeline, because the shortcut is not there.

**A caution about the ResNet embeddings:** the pretrained network may also encode size-related and brightness-related information in its features. A high AUC from the main model is therefore not evidence that it learned pneumonia. Proving that would require the ablations listed in [Future work](#future-work) and testing on external data.

**Bottom line:** the 0.991 AUC should be read as an upper bound on real-world performance, not as an estimate of it.

---

## Project structure

```
cxr-pneumonia/
  cache/
    emb_resnet18.npz        # cached ResNet18 embeddings (not committed)
  data/                     # Kaggle images (not committed)
  reports/
    metrics.json            # all numbers, machine readable
    metrics.md              # human readable report
  scripts/
    make_fake_data.py       # generates synthetic data to test the pipeline
  src/                      # data loading, patient split, embedding, training, evaluation
  .gitignore
  README.md
  requirements.txt
```

`scripts/make_fake_data.py` creates a synthetic dataset with a deliberately planted shortcut. It was used to verify that the pipeline and the shortcut check work **before** running on real data. It is a testing tool only. No result in this README comes from synthetic data.

---

## How to run

### Requirements
- Windows, macOS or Linux
- Python 3.9 or newer
- No GPU needed

### Steps

1. **Get the code** and open the `cxr-pneumonia` folder in your editor.

2. **Download the dataset** from Kaggle (search for "Chest X-Ray Images Pneumonia") and unzip it inside the `data/` folder.

3. **Install dependencies** (run each command on its own):

   ```
   py -m pip install -r requirements.txt
   ```

4. **Train and evaluate:**

   ```
   py -m src.train --data data
   ```

   On macOS or Linux, use `python3` instead of `py`.

5. **Be patient on the first run.** Extracting embeddings takes roughly 10 to 40 minutes on a CPU, and progress prints as `embedded X/5800`. Embeddings are cached, so later runs finish quickly.

6. **Read the results** in `reports/metrics.md` and `reports/metrics.json`.

### Troubleshooting

- **`No module named torch`:** run the install step again, or install PyTorch following the instructions on pytorch.org for your system.
- **Dataset not found:** make sure the unzipped images are somewhere under `data/`. The exact sub-folder layout is not important.
- **Want to recompute embeddings:** delete `cache/emb_resnet18.npz` and run again.
- **`py` is not recognised:** use `python` or `python3`.

---

## Reproducibility notes

- The patient split uses a fixed random seed, so the same patients land in the same groups each run.
- Embeddings are cached. Deleting the cache and re-running should reproduce the same features.
- Confidence intervals come from bootstrap resampling, so the last digit can move slightly if the seed or number of resamples changes.
- Results were produced on the Kaggle dataset described above. A different download or a different cleaning of the data can change the numbers.

---

## Limitations

- **Strong dataset shortcut.** Image size and brightness alone give an AUC of 0.942, so the main score is likely inflated.
- **The ResNet18 is frozen**, so accuracy is lower than a fully fine-tuned model would likely reach.
- **Patient IDs are parsed from Kaggle file names.** This is a good approximation but not a guarantee that every patient is perfectly grouped.
- **Accuracy alone is misleading** because pneumonia is about 72% of the data. Sensitivity, specificity and AUC are more informative.
- **Single source.** All images come from one dataset, so there is no evidence about performance on other hospitals, devices or age groups. The images are pediatric, so results do not transfer to adult chest X-rays.
- **No clinical validity.** This is a learning and research project. It must not be used for diagnosis.

---

## Future work

- Run ablations: shortcut features only, ResNet embeddings only, and both combined. If adding the shortcut features does not improve the ResNet model, that is evidence the embeddings already carry them.
- Remove the shortcut: resize every image to a fixed size, normalise brightness and contrast (for example histogram equalisation), then re-run and see how much the AUC drops.
- Add Grad-CAM to see which image regions drive the predictions. If the heat falls on borders, text or markers instead of the lungs, that confirms shortcut learning.
- Evaluate on an external chest X-ray dataset from a different source.
- Fine-tune the network (requires a GPU) and compare other backbones such as DenseNet121.
- Add calibration analysis and a threshold study, since the 0.5 threshold is not necessarily the right trade-off between missed pneumonia and false alarms.

---

## References

- Kermany, D. et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018. (Origin of the chest X-ray pneumonia dataset.)
- He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016. (ResNet.)
- Selvaraju, R. et al. *Grad-CAM: Visual Explanations from Deep Networks.* ICCV, 2017.
- Geirhos, R. et al. *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence, 2020.
- Zech, J. et al. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs.* PLOS Medicine, 2018.

---

## Disclaimer

This repository is for education and research only. It is not a medical device and must not be used to make clinical decisions.
