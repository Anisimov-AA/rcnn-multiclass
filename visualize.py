import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

        return img, {"boxes": boxes, "labels": labels}


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
print("Model loaded from", MODEL_PATH)

# load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = ObjectDataset(TEST_CSV, TEST_IMG_DIR, transform)
print(f"Loaded {len(test_data)} test images")

# show all 24 test predictions
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(4, 6, figsize=(18, 12))

for i in range(len(test_data)):
    ax = axes[i // 6][i % 6]
    img_tensor, target = test_data[i]

    with torch.no_grad():
        cls_pred, _ = model(img_tensor.unsqueeze(0).to(device))
        pred_label = idx_to_class[cls_pred.argmax(1).item()]
        true_label = idx_to_class[target["labels"][0].item()]

    img = img_tensor.numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    ax.imshow(img)
    color = "green" if pred_label == true_label else "red"
    ax.set_title(f"{pred_label} / {true_label}", color=color, fontsize=9)
    ax.axis("off")

plt.suptitle("All Test Predictions (green=correct, red=wrong)", fontsize=14)
plt.tight_layout()
plt.savefig("screenshots/all_predictions.png")
plt.show()
print("Saved screenshots/all_predictions.png")