"""
Test / Evaluation Script for Offroad Segmentation
Fully aligned with train_segmentation.py
"""

import os
import glob
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# ==========================================================
# CONFIG (MUST MATCH TRAINING)
# ==========================================================

NUM_CLASSES = 6

ID_TO_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    27: 4,
    39: 5
}

CLASS_NAMES = [
    "Class_0",
    "Class_1",
    "Class_2",
    "Class_3",
    "Class_27",
    "Class_39"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# DATASET
# ==========================================================

class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.*"))) if mask_dir else None
        self.transform = transform
        self.length = len(self.img_paths)

    def _map_mask(self, mask):
        mask_mapped = np.zeros_like(mask, dtype=np.int64)
        for original_id, new_id in ID_TO_CLASS.items():
            mask_mapped[mask == original_id] = new_id
        return mask_mapped

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_paths:
            mask = cv2.imread(self.mask_paths[idx])
            mask = mask[:, :, 0]
            mask = self._map_mask(mask)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long(), os.path.basename(self.img_paths[idx])


# ==========================================================
# TRANSFORM (Same as validation)
# ==========================================================

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2(),
])

# ==========================================================
# METRICS (Same logic as training)
# ==========================================================

def compute_iou(hist):
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    return intersection.float() / (union.float() + 1e-6)


# ==========================================================
# COLOR PALETTE FOR VISUALIZATION
# ==========================================================

COLOR_PALETTE = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [255, 0, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
], dtype=np.uint8)


def colorize_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        colored[mask == i] = COLOR_PALETTE[i]
    return colored


# ==========================================================
# MAIN
# ==========================================================

def main():

    IMG_DIR = "../Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val/Color_Images"
    MASK_DIR = "../Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val/Segmentation"

    OUTPUT_DIR = "test_predictions"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "colored"), exist_ok=True)

    dataset = OffroadDataset(IMG_DIR, MASK_DIR, transform=val_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(f"Loaded {len(dataset)} samples")

    # Load model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(torch.load("train_stats/best_model.pth", map_location=DEVICE))
    model.eval()

    print("Model loaded successfully")

    hist = torch.zeros(NUM_CLASSES, NUM_CLASSES).to(DEVICE)

    with torch.no_grad():
        for images, masks, filenames in tqdm(loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Confusion matrix
            mask_flat = masks.view(-1)
            pred_flat = preds.view(-1)
            valid = (mask_flat >= 0) & (mask_flat < NUM_CLASSES)

            hist += torch.bincount(
                NUM_CLASSES * mask_flat[valid] + pred_flat[valid],
                minlength=NUM_CLASSES ** 2
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            # Save predictions
            for i in range(images.size(0)):
                pred_mask = preds[i].cpu().numpy().astype(np.uint8)
                colored_mask = colorize_mask(pred_mask)

                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, filenames[i]),
                    pred_mask
                )

                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, "colored", filenames[i]),
                    cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                )

    # Compute IoU
    iou = compute_iou(hist)
    miou = torch.mean(iou).item()

    print("\n==============================")
    print(f"Mean IoU: {miou:.4f}")
    print("==============================")

    for i, cls_iou in enumerate(iou):
        print(f"{CLASS_NAMES[i]}: {cls_iou.item():.4f}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
        f.write(f"Mean IoU: {miou:.4f}\n\n")
        for i, cls_iou in enumerate(iou):
            f.write(f"{CLASS_NAMES[i]}: {cls_iou.item():.4f}\n")

    print("\nPredictions saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()