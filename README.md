# rcnn-multiclass

- **Architecture:** R-CNN with VGG16 (pretrained) backbone
- **Classes:** 4 (tv, remote, wine_bottle + background)

## Dataset

120 images (40/class), annotated with LabelMe, 80/20 split.

[dataset link]()

## Project Structure

```
rcnn-multiclass/
├── rcnn_multi.py          # main training script
├── visualize.py           # test visualization with bounding boxes
├── json_to_csv.py         # converts LabelMe json to csv
├── split_dataset.py       # train/test split
├── rename_images.py       # batch rename images
├── annotations.csv        # all annotations
├── train.csv              # training split
├── test.csv               # test split
├── models/                # saved model weights
└── screenshots/           # training loss plot, predictions
```
