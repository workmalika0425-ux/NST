# 🎨 Neural Style Transfer with AdaIN

> Real-time artistic style transfer using Adaptive Instance Normalization (AdaIN) — transfer the brushstrokes, texture, and mood of any artwork onto your photos in a single forward pass.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 What Is This?

This project implements **Adaptive Instance Normalization (AdaIN)** for arbitrary neural style transfer, based on the paper [*Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*](https://arxiv.org/abs/1703.06868) (Huang & Belongie, 2017).

Unlike iterative optimization-based methods (which can take minutes per image), AdaIN performs style transfer in a **single forward pass** — making it fast enough for real-world use. The model learns to align the statistical distribution (mean and standard deviation) of content image features to those of a style image, all in the feature space of a pre-trained VGG encoder.

The project ships with:
- A **training pipeline** to train your own decoder
- A **Flask web app** for interactive style transfer in the browser
- A **Jupyter notebook** for feature visualization and analysis

---

## 🧠 How It Works

```
Content Image ──┐
                ├──► VGG Encoder ──► AdaIN ──► Decoder ──► Stylized Image
Style Image ────┘
```

**Three components power this system:**

**1. VGG Encoder** — A pre-trained, frozen VGG-19 network (up to `relu4_1`) acts as a feature extractor. It encodes both the content and style images into deep feature representations.

**2. Adaptive Instance Normalization (AdaIN)** — The core of the method. For each feature channel, it shifts the mean and standard deviation of the content features to match those of the style features:

```
AdaIN(x, y) = σ(y) * ((x − μ(x)) / σ(x)) + μ(y)
```

This single operation transfers the style's statistics onto the content's structure — no iterative optimization needed.

**3. Decoder** — A lightweight, trainable convolutional decoder mirrors the encoder architecture and reconstructs a stylized image from the AdaIN-transformed features.

**Loss Functions during Training:**
- **Content Loss** — MSE between decoder output features and the AdaIN target `t`
- **Style Loss** — MSE between mean and standard deviation of output features vs. style features at multiple VGG layers (`relu1_1` through `relu4_1`)

**Alpha (α) blending** — At inference time, a slider lets you control the stylization strength:
```
t = α * AdaIN(c, s) + (1 − α) * c
```
where `α = 1.0` is full stylization and `α = 0.0` is the original content image.

---

## 🗂️ Project Structure

```
NST_AdaIN/
│
├── train.py                  # Training script for the decoder
├── app.py                    # Flask web application
├── code.ipynb                # Feature visualization notebook
│
├── vgg_normalised.pth        # Pre-trained VGG-19 weights (required)
│
├── utils/
│   ├── models.py             # VGGEncoder and Decoder architecture
│   └── utils.py             # AdaIN, transforms, dataset helpers
│
├── experiment/               # Training outputs (checkpoints, sample images)
│   └── <experiment_name>/
│       ├── decoder_N.pth
│       ├── optimizer_N.pth
│       └── output_N.png
│
├── static/
│   └── uploads/             # Web app upload directory
│
├── templates/
│   └── index.html           # Web app frontend
│
└── examples/                # Sample images for the notebook
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Backbone Encoder | VGG-19 (normalised, pre-trained) |
| Web Framework | Flask + Flask-WTF + Flask-Bootstrap |
| Image Processing | Pillow (PIL), torchvision |
| Training Utilities | tqdm, argparse |
| Visualization | Matplotlib, Jupyter |
| Frontend Forms | WTForms |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision flask flask-wtf flask-bootstrap pillow tqdm wtforms
```

### 1. Download the VGG Weights

Place `vgg_normalised.pth` in the project root (or specify its path via `--vgg`). This is the pre-trained, channel-normalised VGG-19 backbone used as the encoder.

### 2. Prepare Your Data

Organize your training images into two separate directories:

```
content_data/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

style_data/
    ├── artwork1.jpg
    ├── artwork2.jpg
    └── ...
```

Any large dataset of natural photos works for content (e.g., MS-COCO). For style, use a diverse collection of artworks (e.g., WikiArt).

---

## 🏋️ Training

```bash
python train.py \
    --content_dir /path/to/content_data \
    --style_dir   /path/to/style_data \
    --vgg         ./vgg_normalised.pth \
    --experiment  my_experiment \
    --epochs      20 \
    --batch_size  8 \
    --lr          1e-4 \
    --content_weight 1.0 \
    --style_weight   5.0
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--content_dir` | required | Path to content image dataset |
| `--style_dir` | required | Path to style image dataset |
| `--vgg` | required | Path to `vgg_normalised.pth` |
| `--experiment` | `experiment1` | Name of the experiment (saves to `experiment/<name>/`) |
| `--epochs` | `1` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-4` | Adam learning rate |
| `--lr_decay` | `5e-5` | Learning rate decay factor |
| `--content_weight` | `1.0` | Weight for content loss |
| `--style_weight` | `5.0` | Weight for style loss |
| `--final_size` | `256` | Output image resolution |
| `--save_interval` | `2` | Save checkpoint every N epochs |
| `--resume` | `False` | Resume from checkpoint (requires `--decoder_path` and `--optimizer_path`) |

Training checkpoints, optimizer states, and sample output grids are saved to `experiment/<experiment_name>/`.

**Resuming training:**
```bash
python train.py --resume \
    --decoder_path   experiment/my_experiment/decoder_10.pth \
    --optimizer_path experiment/my_experiment/optimizer_10.pth \
    ...
```

---

## 🌐 Web Application

The Flask app provides a browser-based interface to upload content and style images and perform style transfer interactively.

### Setup

Update the decoder path in `app.py` to point to your trained decoder:

```python
decoder.load_state_dict(torch.load('experiment/my_experiment/decoder_final.pth'))
```

### Run the App

```bash
python app.py
```

Then open your browser at [http://localhost:5000](http://localhost:5000).

### Features
- Upload a **content image** (the photo to be styled)
- Upload a **style image** (the artwork to draw style from)
- Adjust the **Alpha (α)** slider to control stylization intensity (0.0 = original, 1.0 = fully stylized)
- Download the resulting stylized image

Supported formats: `.png`, `.jpg`, `.jpeg`

---

## 🔬 Feature Visualization Notebook

`code.ipynb` lets you explore what the VGG encoder "sees" at each layer for original and stylized images.

It visualizes the **mean activation maps** at `relu1_1`, `relu2_1`, `relu3_1`, and `relu4_1` side-by-side for multiple images — useful for understanding how style statistics shift across the network.

To use it, place example images in an `examples/` directory and launch:

```bash
jupyter notebook code.ipynb
```

---

## 📊 Model Architecture

### VGGEncoder
- Uses VGG-19 up to `relu4_1` as a fixed feature extractor
- Returns intermediate activations at `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`
- Weights are frozen during training

### Decoder
- Mirrors the VGG encoder in reverse (upsampling via nearest-neighbor + conv)
- Trained from scratch to reconstruct images from AdaIN-transformed features
- Only component updated during training

---

## 📌 Tips & Notes

- A higher `--style_weight` produces more stylized output; a lower value preserves more content structure.
- The `--final_size` controls the resolution of training output images; increase for sharper results (at the cost of GPU memory).
- For best web app results, use a decoder trained for at least 5–10 epochs on a reasonably large dataset (10k+ images per split).
- The VGG encoder runs in `eval()` mode and its parameters are never updated.

---

## 📄 References

- Huang, X., & Belongie, S. (2017). [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868). ICCV 2017.
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). CVPR 2016.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

This project is licensed under the MIT License.
