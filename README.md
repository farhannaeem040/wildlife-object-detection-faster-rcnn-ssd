# Wildlife Object Detection — Faster R-CNN vs. SSD

An end-to-end deep-learning project that detects and classifies three Sub-Saharan wildlife species in camera-trap-style imagery. Two object-detection architectures — **Faster R-CNN** and **SSD** — are trained on the same custom-labelled dataset and compared on accuracy and inference speed.

## Overview

Working from a labelled subset of the Sub-Saharan Africa wildlife dataset, the pipeline covers everything from manual tagging and dataset preparation through TFRecord generation, transfer learning, evaluation, export and inference. Both models are trained with the **TensorFlow 2 Object Detection API** using pre-trained checkpoints from the TF2 Detection Model Zoo.

## Objectives

- Pre-process a multi-class wildlife image dataset and produce at least 1,000 tagged bounding boxes per chosen class.
- Generate Pascal VOC XML annotations, clean malformed pairs, and produce a 90/10 train/test split.
- Convert the tagged dataset into TFRecord files compatible with the TensorFlow 2 Object Detection API.
- Train two architectures (Faster R-CNN and SSD) using transfer learning from pre-trained checkpoints.
- Monitor and analyse training behaviour with TensorBoard (mAP, loss curves, learning rate).
- Evaluate, export and run inference with both models, then compare them.

## Technologies Used

- Python 3, Jupyter Notebook
- TensorFlow 2 / TensorFlow Object Detection API
- Faster R-CNN and SSD model families (config files supplied by the API)
- Pascal VOC XML annotation format, TFRecord
- TensorBoard for training visualisation
- NumPy, pandas, Matplotlib, OpenCV / PIL for EDA and image handling

## Project Structure

```
wildlife-object-detection-faster-rcnn-ssd/
├── 01_preprocessing.ipynb              # EDA, species selection, augmentation, image-quality checks
├── 02_training.ipynb                   # Dataset prep (XML cleanup, splits, TFRecords) + training launch
├── 03_tensorboard_analysis.ipynb       # TensorBoard analysis of Faster R-CNN and SSD training runs
├── 04_evaluation_and_deployment.ipynb  # Evaluation commands and exported saved_model
├── 05_inference.ipynb                  # Inference on test images with both trained models
├── README.md
└── .gitignore
```

The notebooks are numbered to reflect the intended execution order.

## Installation

```bash
git clone https://github.com/farhannaeem040/wildlife-object-detection-faster-rcnn-ssd.git
cd wildlife-object-detection-faster-rcnn-ssd

# Core stack (use conda or venv as you prefer)
pip install tensorflow numpy pandas matplotlib pillow opencv-python jupyter

# TensorFlow Object Detection API
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
```

You will also need:

- The **Sub-Saharan Africa wildlife image dataset** (`SubSaharan.zip`, supplied via the source module).
- **Pre-trained checkpoints** for Faster R-CNN and SSD from the TF2 Detection Model Zoo.
- An **annotation tool** capable of producing Pascal VOC XML (e.g. labelImg) — used to generate the 1,000+ tags per class outside the notebooks.

> The notebooks contain hard-coded Windows paths (for example `C:\Users\MSC1\Desktop\Tensorflow-Object-Detection-API\...`). Update them to match your machine before running.

## Usage

1. **Tagging.** Tag at least 1,000 bounding boxes per chosen species using labelImg or an equivalent tool, exporting Pascal VOC XMLs alongside the JPEGs.
2. **`01_preprocessing.ipynb`** — review the dataset, plot the species distribution, select the three target species, and inspect image quality.
3. **`02_training.ipynb`** — copy JPG/XML pairs into the working `images` directory, clean malformed XMLs, run `partition_dataset.py` for a 90/10 split, generate `label_map.pbtxt` and TFRecords, then launch training and TensorBoard.
4. **`03_tensorboard_analysis.ipynb`** — read the precision (mAP) and loss curves logged by both models.
5. **`04_evaluation_and_deployment.ipynb`** — run the API's evaluation command and export `saved_model` artefacts for both architectures.
6. **`05_inference.ipynb`** — load each exported model and run inference on held-out test images.

## Implementation Details

- **Species selection.** Three classes are chosen from the Sub-Saharan dataset (the notebook focuses on *Oryx gazella*, *Phacochoerus africanus* and a third species selected during EDA).
- **Annotations.** Pascal VOC XML files are produced via manual tagging; the notebook then strips malformed entries and ensures every XML has a matching JPG before splitting.
- **Splitting and encoding.** `partition_dataset.py` produces a 90/10 train/test split; `generate_tfrecord.py` (from the TF Object Detection API) converts the resulting folders into TFRecords using `label_map.pbtxt`.
- **Training.** Two pipelines are configured — one for a Faster R-CNN model and one for an SSD model — by editing the API's pipeline config files (paths, number of classes, batch size, fine-tune checkpoint). Training is launched with `model_main_tf2.py`.
- **Monitoring.** TensorBoard runs on port 6006. The third notebook captures and discusses precision (`DetectionBoxes_Precision/mAP`), classification/localisation loss curves, and learning-rate behaviour for both architectures.
- **Export and inference.** `exporter_main_v2.py` produces a `saved_model` for each architecture. The inference notebook loads each model, runs detections on test images, and visualises the output bounding boxes with the category index from `label_map.pbtxt`.

## Results

The notebooks produce, for each architecture:

- Training and evaluation curves in TensorBoard (mAP, loss components, learning rate).
- An exported `saved_model` directory.
- Side-by-side inference visualisations on test images.

A qualitative comparison between Faster R-CNN (typically higher accuracy, slower) and SSD (typically faster, smaller) is discussed in the TensorBoard notebook. Specific numeric metrics depend on tagging quality, hardware and training duration — re-run the notebooks to reproduce them on your own setup.

## Notes and Limitations

- **Inference notebook size.** `05_inference.ipynb` contains embedded inference visualisations and is large (~78 MB). Cloning the repo will pull this file in full; GitHub may display a "large file" notice when viewing it.
- **No dataset or model weights in this repo.** Image data, tagged XMLs, TFRecords and trained checkpoints are not included — they are large and license-restricted. Regenerate them from the original dataset following the steps above.
- **Hard-coded paths.** Expect to edit absolute Windows paths in the notebooks for any new environment.
- **GPU required.** Both architectures are GPU-intensive; the notebooks assume access to a CUDA-capable GPU.

## Author

**Farhan Mohammad Naeem** — MSc coursework project, *Deep Learning*.
