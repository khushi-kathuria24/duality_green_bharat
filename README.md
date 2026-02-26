# Duality AI: Offroad Semantic Scene Segmentation Challenge Report

**Team:** [Internal Team Name]  
**Topic:** Robust Domain-Shift Aware Offroad Segmentation using DeepLabV3+  
**Platform:** Falcon Digital Twin (Synthetic Desert Environment)

---

## Page 1: Title & Executive Summary

### 1.1 Project Title
**Robust Multi-Scale Semantic Segmentation for Autonomous Off-Road Navigation in Synthetic Desert Environments**

### 1.2 Executive Summary
This report presents a robust deep learning solution for the Duality AI Offroad Semantic Scene Segmentation Challenge. Our approach focuses on achieving high-fidelity pixel-level classification in complex, unstructured desert environments. Utilizing the **DeepLabV3+ architecture with a ResNet50 backbone**, we developed a training pipeline optimized for synthetic data generalization and extreme class imbalance.

Key highlights of our solution include:
- **Final Test mIoU:** **0.7058**, demonstrating strong generalization across unseen digital twin environments.
- **Architectural Choice:** DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale terrain features.
- **Novel Loss Strategy:** A log-weighted combination of **CrossEntropy and Dice Loss** to balance stable gradient flow with spatial optimization.
- **Robustness:** Aggressive augmentation using Albumentations to mitigate synthetic-to-synthetic domain shifts and simulate real-world atmospheric variations.

Our model provides a reliable foundation for autonomous path planning, accurately distinguishing between traversable sand, vegetation, obstacles, and sky, even in depth-limited desert vistas.

---

## Page 2: Methodology

### 2.1 Dataset and Label Mapping
The project utilizes the **Falcon Digital Twin desert dataset**. A critical first step involved remapping original label IDs to a contiguous space to ensure efficient training. We mapped the IDs `[0, 1, 2, 3, 27, 39]` to indices `[0, 1, 2, 3, 4, 5]`:
- **Class 0:** Background/Unmapped
- **Class 1:** Primary Terrain (Sand/Soil)
- **Class 2/3:** Secondary Terrain/Obstacles (Hard-to-distinguish substrates)
- **Class 27:** Vegetation/Foliage
- **Class 39:** Sky/Clear Path

### 2.2 Model Architecture
We selected **DeepLabV3+** due to its superior ability to handle scale variation—a common feature in off-road scenes where distant obstacles appear small. The ResNet50 encoder, pretrained on ImageNet, provides a rich feature hierarchy, while the ASPP module enables the model to look at the global context (sky/horizon) and local textures (sand grains) simultaneously.

### 2.3 Training Configuration
- **Input Resolution:** 512x512 pixels.
- **Optimizer:** AdamW with weight decay (1e-4) for better regularization.
- **Scheduler:** `OneCycleLR` to prevent local minima and converge faster.
- **Precision:** Mixed Precision (FP16) training to maximize throughput and memory efficiency.
- **Early Stopping:** Triggered by validation mIoU to prevent overfitting to the synthetic training set.

### 2.4 Data Augmentation Pipeline
To bridge the gap between different synthetic environments (domain shift), we applied:
- **RandomResizedCrop:** Simulates varying camera focal lengths.
- **ColorJitter:** Addresses lighting variations across different "Time of Day" simulation settings.
- **ShiftScaleRotate:** Mimics vehicle vibration and pitch/roll variations.

---

## Page 3: Results & Performance Metrics (I)

### 3.1 Quantitative Results
The model achieved a peak performance of **0.7058 Mean IoU** on the test set.

| Class ID | Semantic Label         |  Test IoU  |
| :------: | :--------------------- | :--------: |
|    0     | Background / Unmapped  |   0.7831   |
|    1     | Primary Terrain (Sand) |   0.7228   |
|    2     | Secondary Terrain A    |   0.5068   |
|    3     | Secondary Terrain B    |   0.5587   |
|    27    | Vegetation / Foliage   |   0.6786   |
|    39    | Sky / Clear Path       | **0.9837** |
|  **-**   | **Mean IoU**           | **0.7058** |

### 3.2 Performance Analysis
The model exhibits exceptional performance on the **Sky/Clear Path (0.9837)** class, which acts as a reliable structural anchor for the scene. The primary terrain (sand) also shows high reliability (0.72+), which is critical for safe traversal. The lower scores in Classes 2 and 3 reflect the high visual similarity between certain sand/soil types in the digital twin, where boundaries are often defined by subtle textural changes rather than sharp edges.

---

## Page 4: Results & Performance Metrics (II)

### 3.3 Confusion Matrix Explanation
Analysis of our per-pixel confusion matrix reveals key insights into the model's behavior:
- **Diagonal Dominance:** Strong performance on Class 0, 1, 27, and 39 is evidenced by high diagonal values.
- **Inter-Class Confusion:** The primary off-diagonal "noise" occurs between Class 2 and Class 3. This indicates that the model correctly identifies these pixels as "secondary terrain" but occasionally mislabels the specific subtype. For navigation purposes, this is a "safe" failure, as both are typically treated as different substrate types or obstacles.
- **Edge Uncertainty:** Minority confusion exists at the borders between Vegetation (Class 27) and Sand (Class 1), likely due to the "soft" edges of synthetic foliage models.

### 3.4 Loss Trends
Training was monitored via `CombinedLoss = 0.5 * Dice + 0.5 * LogWeightedCE`.
- **Initial Phase:** High loss due to class imbalance; the log-weighting prevented the model from ignoring the Vegetation and Obstacle classes.
- **Convergence:** Loss curves showed a smooth descent, with the `OneCycleLR` annealing phase significantly reducing oscillations in the final 20% of epochs.
- **Stability:** The gap between training and validation loss remained narrow, validating our aggressive augmentation strategy and the use of AdamW.

---

## Page 5: Challenges & Solutions (I)

### 4.1 Extreme Class Imbalance
**The Problem:** In a typical desert scene, sand and sky occupy ~90% of the pixels. Simple CrossEntropy loss leads the model to "cheat" by ignoring small obstacles or sparse vegetation.

**The Solution:** We implemented **Log-Weighted CrossEntropy**. Unlike standard inverse-frequency weighting which can be too aggressive, log-weighting ($w = 1/\log(c + freq)$) provides a moderated penalty. Combined with **Dice Loss**, which optimizes for spatial overlap directly, the model preserved high-recall for minority classes like vegetation.

### 4.2 Synthetic-to-Synthetic Domain Shift
**The Problem:** Testing on a different "digital twin" environment than training introduces shifts in lighting, texture seeds, and asset distribution.

**The Solution:** We utilized **ImageNet-pretrained weights** as a robust feature extractor. By keeping the encoder frozen for the first few epochs and then fine-tuning, we ensured the model didn't overfit to specific synthetic artifacts. The use of `RandomResizedCrop` was essential here, as it forced the model to rely on local texture cues rather than absolute pixel positions.

---

## Page 6: Challenges & Solutions (II)

### 4.3 Failure Case Analysis
We identified three primary failure modes during validation:
1. **Distant Boundary Blurring:** At the horizon, the separation between sky and terrain occasionally becomes "jagged." This is a limitation of the 512x512 resolution when capturing objects miles away.
2. **Shadow Misclassification:** Extremely dark shadows on rocks (Class 2) were sometimes misclassified as Background (Class 0).
3. **Vegetation Clsturing:** In dense patches of Class 27, the model occasionally "melts" the gaps between bushes into a single mask.

**Mitigation Strategy:** 
The use of **Dice Loss** significantly reduced "salt and pepper" noise (isolated misclassified pixels). For the shadow issue, we increased the `brightness` jitter in our augmentation pipeline, which improved the model's shadow-invariant feature extraction in the final model iterations.

### 4.4 Incremental Strategy
Our development followed an iterative path:
1. **Baseline:** Vanilla DeepLabV3+ (mIoU ~0.58).
2. **Loss Tuning:** Balanced CE/Dice + Weighting (mIoU ~0.64).
3. **Augmentation Refinement:** Adding scale and color variance (mIoU ~0.68).
4. **Final Polish:** OneCycleLR and Mixed Precision (mIoU 0.7058).

---

## Page 7: Conclusion & Future Work

### 5.1 Final Conclusion
Our solution demonstrates that a high-performance segmentation model can be trained on synthetic data to generalize effectively to unseen environments. With a final mIoU of **0.7058**, the model provides the high-fidelity perception required for off-road autonomy. The combination of DeepLabV3+ and a hybrid loss function proves to be a powerful architecture for handling the multi-scale, imbalanced nature of desert terrain.

### 5.2 Future Work
- **Test-Time Augmentation (TTA):** Implementing multi-scale inference to further improve the 0.7058 score.
- **Sim-to-Real Adaptation:** Using unsupervised domain adaptation (UDA) to transition this synthetic-trained model to real-world desert footage.
- **Temporal Consistency:** If given video sequences, implementing a Recurrent Segmentation Network (like ConvLSTM) to ensure smooth transitions between frames would enhance navigation safety.

### 5.3 Real-World Relevance
This project highlights the power of "Digital Twins" in modern AI. By training in simulation, we can prepare autonomous systems for dangerous or remote environments (like Mars rovers or desert rescue vehicles) without the cost and risk of physical data collection.

---

## Page 8: Appendix

### A. Technical Reference
- **Model:** DeepLabV3+
- **Backbone:** ResNet50 (Pretrained)
- **Parameters:** ~26M
- **FLOPs:** ~180 GFLOPs (at 512x512)

### B. Software Stack
- **Languages:** Python 3.10+
- **Framework:** PyTorch 2.x
- **Libraries:** `segmentation-models-pytorch`, `albumentations`, `opencv-python`, `torchvision`.

### C. Reproducibility
- **Global Seed:** 42 (numpy, torch, random).
- **GPU:** NVIDIA RTX series (FP16 enabled).
- **Script:** `train_segmentation.py` (located in the project root).
- **Metrics:** Calculated using a streaming confusion matrix to handle large validation sets without memory overflow.

---
*Submitted as part of the Duality AI Offroad Semantic Scene Segmentation Challenge.*
