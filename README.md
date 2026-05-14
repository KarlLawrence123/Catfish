# 🐟 Catfish Disease Classifier (YOLO11s)

Real-time catfish disease detection using YOLO11s with dual-camera monitoring support.

## 🎯 Features

- **7 Disease Classes**: Bacterial Gill Disease, Columnaris, Healthy, MAS, Saprolegniasis, Unhealthy, White Spot
- **YOLO11s Model**: 9.4M parameters, optimized for Raspberry Pi 5 deployment
- **Full Augmentation Suite**: Underwater-aware augmentations (hsv, perspective, mixup, copy-paste)
- **Roboflow-Style Training Graphs**: Precision, Recall, F1, mAP@50, mAP@50-95, F1-Confidence curves
- **Resume Training**: Automatic checkpointing every 50 epochs
- **NCNN Export**: Optimized for edge deployment on Raspberry Pi 5

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
# Fresh training
python train_catfish.py

# Resume from checkpoint (if interrupted)
python train_catfish.py --resume
```

### 3. Results
After training completes:
- **Model**: `runs/detect/catfish_v7_SMALL_final/weights/best.pt`
- **Graphs**: `runs/detect/catfish_v7_SMALL_final/training_graphs_roboflow.png`
- **NCNN Export**: `runs/detect/catfish_v7_SMALL_final/weights/best.ncnn`

## 📊 Training Performance

The model includes comprehensive metrics:
- Precision/Recall/F1 per epoch
- mAP@50 and mAP@50-95
- Loss curves (box, classification, DFL)
- F1-Confidence curve for optimal threshold selection

## 🎮 Dataset

- **Source**: Roboflow Catfish Disease Dataset v6
- **Classes**: 7
- **Images**: 6,320 train, 279 validation
- **Auto-download**: Dataset downloaded automatically during training

## 📱 Deployment

### Raspberry Pi 5
```bash
# Use the exported NCNN model
# Model: best.ncnn (FP16 optimized)
# Size: 640x640
# Expected FPS: 10-15 on Pi 5
```

## 🛠️ Hardware Requirements

- **Training**: NVIDIA GPU with 8GB+ VRAM (RTX 3060/4060)
- **Inference**: CPU or GPU
- **Storage**: ~2GB for dataset + models

## 📈 Key Improvements

- **Underwater Augmentations**: hsv_v, perspective, shear for water conditions
- **Advanced Augmentations**: mixup, copy-paste for rare disease cases
- **Classification Gain**: cls=1.5 to penalize mislabels
- **Box Gain**: box=7.5 for precise lesion boundaries
- **Learning Rate**: Cosine annealing with warmup

## 📁 Project Structure

```
catfish_disease_classifier/
├── train_catfish.py          # Main training script
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── .gitignore               # Git exclusions
└── runs/detect/             # Training outputs
    └── catfish_v7_SMALL_final/
        ├── weights/
        │   ├── best.pt      # Best model
        │   ├── last.pt      # Last checkpoint
        │   └── best.ncnn    # NCNN export
        └── training_graphs_roboflow.png
```

## 🤖 Model Architecture

- **Base**: YOLO11s (9.4M parameters)
- **Backbone**: CSPDarknet with C3k2 blocks
- **Neck**: PANet with C2PSA attention
- **Head**: Detect head for 7 classes
- **GFLOPs**: 21.6

## 📊 Training Configuration

- **Epochs**: 300 (with early stopping)
- **Batch Size**: 16
- **Image Size**: 640x640
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 with cosine annealing
- **Patience**: 30 epochs

## 🎯 Use Cases

- **Aquaculture Monitoring**: Real-time disease detection
- **Research**: Automated fish health assessment
- **Education**: Computer vision for agriculture
- **Edge AI**: Raspberry Pi deployment

## 📝 License

This project uses the CC BY 4.0 dataset license.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Training typically takes 2-4 hours on GPU. Use `--resume` if interrupted!**
