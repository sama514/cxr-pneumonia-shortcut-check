# Results: chest X-ray NORMAL vs PNEUMONIA

Embedder: `resnet18` | seed: 42 | chosen C (by validation AUC): 0.1 | validation AUC: 0.990

## Split sizes (images / patients)

| Split | NORMAL | PNEUMONIA | Patients |
|---|---|---|---|
| train | 1092 | 2980 | 1952 |
| val | 244 | 651 | 418 |
| test | 247 | 642 | 420 |

## Test-set performance

| Model | AUC | Accuracy | Sensitivity | Specificity | TP/FN/TN/FP |
|---|---|---|---|---|---|
| Main model (threshold 0.5) | 0.991 | 0.963 | 0.970 | 0.943 | 623/19/233/14 |
| Main model (threshold 0.804 chosen on validation) | 0.991 | 0.949 | 0.945 | 0.960 | 607/35/237/10 |
| Shortcut model (size + brightness only) | 0.942 | 0.874 | 0.931 | 0.725 | 598/44/179/68 |

Main model AUC 95% CI (patient-level bootstrap): 0.986 to 0.996
Shortcut model AUC 95% CI: 0.924 to 0.959

## How to read the shortcut check

The shortcut model never sees lung content, only image size and overall brightness.
- AUC near 0.5: no obvious shortcut from these simple properties.
- AUC clearly above 0.5: the dataset leaks the label through acquisition properties, so part of the main model's score may not come from the lungs.
- Compare it with the main model AUC. If they are close, be suspicious of the main result.
