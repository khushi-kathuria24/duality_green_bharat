import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from collections import Counter
import math
import ttach as tta

plt.switch_backend('Agg')

# ==========================================
# 1. Reproducibility
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ==========================================
# 2. Configuration
# ==========================================
class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    TRAIN_IMG_DIR = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset',
                                 'Offroad_Segmentation_Training_Dataset', 'train', 'Color_Images')

    TRAIN_MASK_DIR = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset',
                                  'Offroad_Segmentation_Training_Dataset', 'train', 'Segmentation')

    VAL_IMG_DIR = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset',
                               'Offroad_Segmentation_Training_Dataset', 'val', 'Color_Images')

    VAL_MASK_DIR = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset',
                                'Offroad_Segmentation_Training_Dataset', 'val', 'Segmentation')

    # REAL LABEL IDS FOUND IN DATASET
    LABEL_IDS = [0, 1, 2, 3, 27, 39]

    # Map to contiguous indices [0-5]
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

    NUM_CLASSES = 6

    IMG_SIZE = (512, 512)
    BATCH_SIZE = 8
    EPOCHS = 90
    MAX_LR = 3e-4
    PATIENCE = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# ==========================================
# 3. Class Weight Calculation
# ==========================================
def calculate_class_weights(mask_dir):
    print("Calculating class frequencies...")
    mask_paths = glob.glob(os.path.join(mask_dir, "*.*"))
    class_counts = Counter()

    sample_size = min(500, len(mask_paths))
    sampled = random.sample(mask_paths, sample_size)

    for path in tqdm(sampled):
        # Read as color and use channel 0 (masks are 3-channel with identical channels)
        mask = cv2.imread(path)
        mask = mask[:, :, 0]

        for original_id, new_id in config.ID_TO_CLASS.items():
            class_counts[new_id] += np.sum(mask == original_id)

    total_pixels = sum(class_counts.values())
    weights = []

    print("\nClass Frequencies & Weights:")
    for class_idx in range(config.NUM_CLASSES):
        count = class_counts.get(class_idx, 0)
        freq = count / total_pixels if total_pixels > 0 else 0
        weight = 1.0 / math.log(1.02 + freq)
        weights.append(weight)

        print(f"{config.CLASS_NAMES[class_idx]}: "
              f"Freq={freq*100:.2f}% | Weight={weight:.4f}")

    return weights

# ==========================================
# 4. Dataset
# ==========================================
class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.*")))
        self.transform = transform
        self.length = min(len(self.img_paths), len(self.mask_paths))

    def __len__(self):
        return self.length

    def _map_mask(self, mask):
        mask_mapped = np.zeros_like(mask, dtype=np.int64)
        for original_id, new_id in config.ID_TO_CLASS.items():
            mask_mapped[mask == original_id] = new_id
        return mask_mapped

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read as color and extract channel 0 for accurate label values
        mask = cv2.imread(self.mask_paths[idx])
        mask = mask[:, :, 0]
        mask = self._map_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()

# ==========================================
# 5. Augmentations
# ==========================================
train_transform = A.Compose([
    A.RandomResizedCrop(height=512, width=512, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(),
    ToTensorV2(),
])

# ==========================================
# 6. Metrics
# ==========================================
def compute_iou(hist):
    intersection = torch.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    return intersection.float() / (union.float() + 1e-6)

# ==========================================
# 7. Combined Loss
# ==========================================
class CombinedLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + 0.5 * self.dice(pred, target)

# ==========================================
# 8. Training
# ==========================================
def main():
    os.makedirs("train_stats", exist_ok=True)

    weights = calculate_class_weights(config.TRAIN_MASK_DIR)
    weights_tensor = torch.tensor(weights).float().to(config.DEVICE)

    train_ds = OffroadDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, train_transform)
    val_ds = OffroadDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, val_transform)

    # drop_last=True prevents last batch with 1 sample from crashing BatchNorm
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=config.NUM_CLASSES
    ).to(config.DEVICE)

    criterion = CombinedLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.MAX_LR, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=config.MAX_LR,
                           epochs=config.EPOCHS,
                           steps_per_epoch=len(train_loader))

    scaler = torch.amp.GradScaler("cuda")

    best_miou = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_miou': []}

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        # TRAIN
        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATE
        model.eval()
        hist = torch.zeros(config.NUM_CLASSES, config.NUM_CLASSES).to(config.DEVICE)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                with torch.amp.autocast("cuda"):
                    outputs = model(images)

                preds = torch.argmax(outputs, 1)

                # Vectorized confusion matrix (FAST)
                mask_flat = masks.view(-1)
                pred_flat = preds.view(-1)
                valid = (mask_flat >= 0) & (mask_flat < config.NUM_CLASSES)
                hist += torch.bincount(
                    config.NUM_CLASSES * mask_flat[valid] + pred_flat[valid],
                    minlength=config.NUM_CLASSES ** 2
                ).reshape(config.NUM_CLASSES, config.NUM_CLASSES)

        iou = compute_iou(hist)
        miou = torch.mean(iou).item()

        history['train_loss'].append(train_loss)
        history['val_miou'].append(miou)

        print(f"Train Loss: {train_loss:.4f} | Val mIoU: {miou:.4f}")
        for i, cls_iou in enumerate(iou):
            print(f"  {config.CLASS_NAMES[i]}: IoU = {cls_iou.item():.4f}")

        lowest_idx = torch.argmin(iou).item()
        print(f"  Lowest: {config.CLASS_NAMES[lowest_idx]} (IoU: {iou[lowest_idx]:.4f})")

        # Early Stopping & Checkpointing
        if miou > best_miou:
            best_miou = miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), "train_stats/best_model.pth")
            print("Best model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= config.PATIENCE:
                print("Early stopping triggered!")
                break

    # ==========================================
    # 9. Save Training Curves & Metrics
    # ==========================================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_miou'], label='Val mIoU', color='blue')
    plt.title('Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('train_stats/training_curves.png')
    plt.close()
    print("Saved training_curves.png")

    with open('train_stats/final_metrics.txt', 'w') as f:
        f.write(f"Best Validation mIoU: {best_miou:.4f}\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Val mIoU: {history['val_miou'][-1]:.4f}\n\n")
        f.write("Per-class IoU (final epoch):\n")
        for i, cls_iou in enumerate(iou):
            f.write(f"  {config.CLASS_NAMES[i]}: {cls_iou.item():.4f}\n")
    print("Saved final_metrics.txt")

if __name__ == "__main__":
    main()