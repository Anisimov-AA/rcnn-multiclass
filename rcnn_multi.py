import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# paths - change these if running locally
DATA_DIR = "/kaggle/input/datasets/tylerde/rcnn-multiclass-dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "dataset", "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "dataset", "test")

# training settings from assignment
num_classes = 4  # tv, remote, wine_bottle + background
batch_size = 2
epochs = 10
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# maps class names to numbers and back
class_to_idx = {"tv": 1, "remote": 2, "wine_bottle": 3}
idx_to_class = {0: "background", 1: "tv", 2: "remote", 3: "wine_bottle"}


# custom dataset class to load images and bounding boxes from csv
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


# image preprocessing - resize to 224x224 for VGG16
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# need custom collate because each image can have different number of boxes
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


# load data
train_data = ObjectDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform)
test_data = ObjectDataset(TEST_CSV, TEST_IMG_DIR, transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
print(f"Loaded {len(train_data)} train and {len(test_data)} test images")


# R-CNN model using VGG16 as backbone
# has two heads: one for classification, one for bounding box regression
class SimpleRCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.backbone = vgg.features
        self.pool = vgg.avgpool

        # freeze first few layers so they dont change during training
        for p in self.backbone[:10].parameters():
            p.requires_grad = False

        # classification head - predicts which class
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes)
        )

        # bbox head - predicts bounding box coordinates
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


# setup model, loss functions, optimizer
model = SimpleRCNN(num_classes).to(device)
cls_loss_fn = nn.CrossEntropyLoss()
box_loss_fn = nn.SmoothL1Loss()
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr, momentum=0.9, weight_decay=0.0005
)


# training loop
print("\nTraining...")
loss_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    count = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)

        # grab first annotation per image (we have 1 object per image anyway)
        cls_targets = torch.tensor([t["labels"][0] for t in targets]).to(device)
        box_targets = torch.stack([t["boxes"][0] for t in targets]).to(device)
        # normalize bbox coords to 0-1 range based on max value in batch
        box_max = box_targets.max()
        if box_max > 0:
            box_targets = box_targets / box_max

        cls_pred, box_pred = model(imgs)

        # clamp predictions to avoid nan
        box_pred = torch.clamp(box_pred, -10, 10)

        loss = cls_loss_fn(cls_pred, cls_targets) + box_loss_fn(box_pred, box_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    avg = running_loss / count
    loss_history.append(avg)
    print(f"  Epoch {epoch+1}/{epochs} - loss: {avg:.4f}")


# plot training loss
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
print("Saved training_loss.png")


# test the model
print("\nTesting...")
model.eval()
correct = 0
total = 0
predictions = []

with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        cls_pred, _ = model(imgs)

        pred = cls_pred.argmax(dim=1).item()
        actual = targets[0]["labels"][0].item()

        correct += (pred == actual)
        total += 1
        predictions.append({"actual": idx_to_class[actual], "predicted": idx_to_class[pred]})

print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

# per-class breakdown
print("\nPer class:")
for cls_name in ["tv", "remote", "wine_bottle"]:
    cls_preds = [p for p in predictions if p["actual"] == cls_name]
    cls_correct = sum(1 for p in cls_preds if p["actual"] == p["predicted"])
    print(f"  {cls_name}: {cls_correct}/{len(cls_preds)}")


# show ALL test predictions (24 images: 4 rows x 6 cols)
cols = 6
rows = 4
fig, axes = plt.subplots(rows, cols, fig

# save trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/rcnn_model.pth")
print("Model saved to models/rcnn_model.pth")