# GenderRecognition
One important note before the README: I’d frame this as a **binary gender-classification experiment**, not as a system that can determine someone's actual gender from an image. The model is learning the labels present in the dataset, which is a much more accurate description of what the code does.

# 👤 Gender Recognition CNN

A binary image-classification project built with **PyTorch** that predicts the dataset's **Female/Male** labels from an image.

The project includes a custom CNN architecture with residual blocks, Squeeze-and-Excitation modules, data augmentation, mixed-precision training, cosine learning-rate scheduling, and a Streamlit interface for image inference.

> **Important:** This model predicts the labels it was trained on. It should not be interpreted as determining a person's actual gender or identity from an image.

---

## Overview

The model takes an image of a person and classifies it into one of two dataset labels:

```text
Image
  ↓
128 × 128 preprocessing
  ↓
Custom CNN
  ↓
Feature extraction
  ↓
Global Average Pooling
  ↓
Dropout
  ↓
Linear Classifier
  ↓
Female / Male
```

The Streamlit frontend provides a simple interface for uploading an image and displaying the model's predicted class.

---

## Classes

The model has two output classes:

```python
class_names = ["Female", "Male"]
```

These correspond to the labels provided by the training dataset.

---

# Model Architecture

The CNN was designed from scratch in PyTorch rather than using a pretrained image-classification model.

The architecture consists of five convolutional residual blocks:

```text
Input: 3 × 128 × 128
        │
        ▼
Block 1: 3 → 32
        │
        ▼
Block 2: 32 → 64
        │
        ▼
Block 3: 64 → 128
        │
        ▼
Squeeze-and-Excitation
        │
        ▼
Block 4: 128 → 256
        │
        ▼
Squeeze-and-Excitation
        │
        ▼
Block 5: 256 → 512
        │
        ▼
Squeeze-and-Excitation
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout
        │
        ▼
Linear: 512 → 2
        │
        ▼
Class Prediction
```

---

# Residual Blocks

Each convolutional block contains two convolutional layers with BatchNorm and SiLU activation.

```text
Input
 │
 ├─────────────── Skip Connection ──────────────┐
 │                                               │
 ▼                                               │
3×3 Convolution                                  │
 │                                               │
 ▼                                               │
BatchNorm                                         │
 │                                               │
 ▼                                               │
SiLU                                               │
 │                                               │
 ▼                                               │
3×3 Convolution                                  │
 │                                               │
 ▼                                               │
BatchNorm                                         │
 │                                               │
 └─────────────────── + ─────────────────────────┘
                         │
                         ▼
                       SiLU
```

When the input and output dimensions differ, the skip connection uses a `1×1` convolution with the appropriate stride.

This allows the network to change feature dimensions while still maintaining a residual pathway.

---

# Squeeze-and-Excitation

The model also uses **Squeeze-and-Excitation (SE)** modules after several convolutional blocks.

The SE module learns channel-wise importance weights.

```text
Feature Maps
     │
     ▼
Adaptive Average Pooling
     │
     ▼
Channel Reduction
     │
     ▼
SiLU
     │
     ▼
Channel Expansion
     │
     ▼
Sigmoid
     │
     ▼
Channel Weights
     │
     ×
     │
Feature Maps
```

The SE modules are applied to the:

```text
128-channel features
256-channel features
512-channel features
```

with a reduction ratio of:

```text
r = 16
```

This gives the network a mechanism for emphasizing useful feature channels.

---

# Input Processing

Images are resized to:

```text
128 × 128
```

and normalized using:

```text
Mean = [0.5, 0.5, 0.5]
Std  = [0.5, 0.5, 0.5]
```

For training, several augmentations are applied.

### Training Augmentation

```python
RandomResizedCrop(128, scale=(0.8, 1.0))
RandomHorizontalFlip(p=0.5)
RandomRotation(10)
ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.02
)
```

The validation/test images are only resized and normalized.

This prevents random augmentation from affecting evaluation.

---

# Training

The model was trained for:

```text
15 epochs
```

using:

```text
Optimizer: AdamW
Learning Rate: 1 × 10⁻⁴
Weight Decay: 1 × 10⁻⁴
Batch Size: 64
Loss: Cross Entropy
```

The learning rate was scheduled using cosine annealing:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs
)
```

---

# Mixed Precision

Training uses PyTorch automatic mixed precision:

```python
with autocast(device_type=device):
    outputs = net(inputs)
    loss = loss_function(outputs, labels)
```

Gradient scaling is used to maintain numerical stability:

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This was used to make GPU training more memory-efficient and potentially faster.

---

# Dataset

The project uses an image dataset organized with `torchvision.datasets.ImageFolder`.

The directory structure is expected to follow the standard `ImageFolder` format:

```text
GenderDataset/
│
├── Train/
│   ├── Female/
│   └── Male/
│
└── Test/
    ├── Female/
    └── Male/
```

`ImageFolder` automatically assigns class indices based on the directory names.

The mapping is also stored alongside the trained model checkpoint.

---

# Evaluation

After training, the model is evaluated on the test dataset.

For each image:

```text
Image
  ↓
CNN
  ↓
2 Output Logits
  ↓
Highest Logit
  ↓
Predicted Class
```

The final accuracy is calculated as:

```text
Correct Predictions
──────────────────── × 100
Total Predictions
```

---

# Model Checkpoint

The trained model is saved together with the class mapping:

```python
model_data = {
    "model_state_dict": net.state_dict(),
    "class_to_idx": train_data.class_to_idx
}
```

The resulting checkpoint is:

```text
genderecog_model.pth
```

This allows the trained weights to be loaded later without retraining the network.

---

# Streamlit Frontend

The project includes a simple Streamlit application for inference.

The user can upload:

```text
.png
.jpg
.jpeg
```

images.

The application then:

1. Loads the uploaded image.
2. Converts it to RGB.
3. Resizes it to `128 × 128`.
4. Applies the same normalization used during testing.
5. Passes it through the CNN.
6. Selects the highest-scoring class.
7. Displays the prediction.

```text
Upload Image
     ↓
PIL Image
     ↓
RGB
     ↓
Transform
     ↓
CNN
     ↓
Prediction
     ↓
Streamlit UI
```

---

# Technologies

* Python
* PyTorch
* Torchvision
* Streamlit
* PIL
* CUDA
* NumPy

---

# What I Learned

This was one of my earlier computer vision projects and built on the CNN work I had already done.

The project gave me experience with:

* Designing CNN architectures
* Residual connections
* Squeeze-and-Excitation
* Batch normalization
* SiLU activation
* Global average pooling
* Dropout
* Image augmentation
* Binary image classification
* Cross-entropy loss
* AdamW
* Cosine learning-rate scheduling
* Mixed-precision training
* PyTorch model checkpointing
* Building a simple ML inference application with Streamlit

One of the most useful parts of the project was learning how different architectural components can be combined rather than relying entirely on standard pretrained architectures.

---

# Project Structure

```text
Gender-Recognizer/
│
├── model.py
├── app.py
├── genderecog_model.pth
└── README.md
```

---

# Running the Project

Install the required packages:

```bash
pip install torch torchvision streamlit pillow
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

Upload an image and the model will return one of its two trained dataset labels.

---

# Limitations

This project has several important limitations.

### Dataset Bias

The model can only learn patterns represented in its training dataset. Its predictions may therefore reflect biases or limitations in the dataset.

### Binary Labels

The model was trained only on two labels:

```text
Female
Male
```

It therefore cannot represent the full range of human gender identities.

### Image-Based Prediction

The model predicts a dataset label from visual information. It cannot determine someone's actual gender identity from an image.

### Generalization

Performance on images that differ significantly from the training data may be substantially worse than performance on the test set.

---

# Future Improvements

Possible improvements include:

* Evaluate the model on a more diverse dataset
* Add confidence scores
* Display top-k predictions
* Add model interpretability with Grad-CAM
* Improve dataset balancing
* Compare against pretrained architectures
* Add a proper evaluation report with precision, recall, and F1
* Test robustness on different image conditions
* Improve the frontend

---

## Author

Built in **PyTorch** as an early computer vision project focused on learning CNN architecture design, image augmentation, model training, and deploying a trained model through Streamlit.
