import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# paths
DATA_DIR = "/kaggle/input/datasets/tylerde/rcnn-multiclass-dataset"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "dataset", "test")
MODEL_PATH = "models/rcnn_model.pth"

num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_idx = {"tv": 1, "remote": 2, "wine_bottle": 3}
idx_to_class = {0: "background", 1: "tv", 2: "remote", 3: "wine_bottle"}
class_colors = {"tv": "lime", "remote": "cyan", "wine_bottle": "magenta"}


class ObjectDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.images = self.df["filename"].unique()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        orig_w, orig_h = img.size

        rows = self.df[self.df["filename"] == fname]
        boxes = []
        labels = []
        for _, r in rows.iterrows():
            boxes.append([r["xmin"], r["ymin"], r["xmax"], r["ymax"]])
            labels.append(class_to_idx[r["class"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        return img, {"boxes": boxes, "labels": labels, "orig_size": (orig_w, orig_h), "filename": fname}


class SimpleRCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.backbone = vgg.features
        self.pool = vgg.avgpool

        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes)
        )

        self.box_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.cls_head(x), self.box_head(x)


# load model
model = SimpleRCNN(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = ObjectDataset(TEST_CSV, TEST_IMG_DIR, transform)
print(f"{len(test_data)} test images")

# show all test images with bounding boxes
fig, axes = plt.subplots(4, 6, figsize=(20, 14))

for i in range(len(test_data)):
    ax = axes[i // 6][i % 6]
    img_tensor, target = test_data[i]
    orig_w, orig_h = target["orig_size"]

    with torch.no_grad():
        cls_pred, _ = model(img_tensor.unsqueeze(0).to(device))
        pred_label = idx_to_class[cls_pred.argmax(1).item()]
        true_label = idx_to_class[target["labels"][0].item()]

    # load original image for display (not the normalized one)
    fname = target["filename"]
    orig_img = Image.open(os.path.join(TEST_IMG_DIR, fname)).convert("RGB")
    ax.imshow(orig_img)

    # draw ground truth bounding box
    box = target["boxes"][0]
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    color = class_colors.get(true_label, "white")
    rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)

    # label with prediction
    title_color = "green" if pred_label == true_label else "red"
    ax.set_title(f"pred: {pred_label}", color=title_color, fontsize=8)
    ax.text(xmin, ymin - 5, true_label, color=color, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    ax.axis("off")

plt.suptitle("Test Results - Bounding Boxes + Predictions (green=correct, red=wrong)", fontsize=13)
plt.tight_layout()
os.makedirs("screenshots", exist_ok=True)
plt.savefig("screenshots/all_predictions.png", dpi=150)
plt.show()
print("Saved screenshots/all_predictions.png")