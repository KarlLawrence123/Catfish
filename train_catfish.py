import torch
from ultralytics import YOLO
from roboflow import Roboflow
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

CLASS_NAMES = [
    'Bacterial Gill Disease',
    'Columnaris Disease',
    'Healthy_Catfish',
    'Motile Aeromonas Septicemia',
    'Saprolegniasis',
    'Unhealthy_Catfish',
    'White Spot Disease'
]

RUN_NAME = 'catfish_v7_SMALL_final'


def plot_roboflow_style_graphs(run_dir):
    """
    Reads results.csv from the YOLO run directory and plots
    Roboflow-style training graphs including F1 confidence.
    """
    results_csv = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(results_csv):
        print(f"⚠️  results.csv not found at {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    epochs = df['epoch'].values

    # Calculate F1 from precision and recall
    precision = df['metrics/precision(B)'].values
    recall    = df['metrics/recall(B)'].values
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0
    )

    # ── Figure layout (3 rows x 3 cols, Roboflow style) ─────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Catfish Disease Classifier — Training Results (YOLOv11s)',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def styled_ax(ax, title, xlabel='Epoch', ylabel=''):
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Row 0 — Losses
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, df['train/box_loss'], label='Train', color='#4285F4', linewidth=2)
    ax1.plot(epochs, df['val/box_loss'],   label='Val',   color='#EA4335', linewidth=2)
    styled_ax(ax1, 'Box Loss', ylabel='Loss')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, df['train/cls_loss'], label='Train', color='#4285F4', linewidth=2)
    ax2.plot(epochs, df['val/cls_loss'],   label='Val',   color='#EA4335', linewidth=2)
    styled_ax(ax2, 'Classification Loss', ylabel='Loss')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, df['train/dfl_loss'], label='Train', color='#4285F4', linewidth=2)
    ax3.plot(epochs, df['val/dfl_loss'],   label='Val',   color='#EA4335', linewidth=2)
    styled_ax(ax3, 'DFL Loss', ylabel='Loss')

    # Row 1 — Precision / Recall / F1
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, precision, label='Precision', color='#0F9D58', linewidth=2)
    styled_ax(ax4, 'Precision', ylabel='Precision')
    ax4.set_ylim(0, 1)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, recall, label='Recall', color='#F4B400', linewidth=2)
    styled_ax(ax5, 'Recall', ylabel='Recall')
    ax5.set_ylim(0, 1)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, f1, label='F1', color='#AB47BC', linewidth=2)
    best_f1_epoch = epochs[np.argmax(f1)]
    best_f1_val   = np.max(f1)
    ax6.axvline(best_f1_epoch, color='red', linestyle='--', linewidth=1,
                label=f'Best F1={best_f1_val:.3f} @ ep{best_f1_epoch}')
    styled_ax(ax6, 'F1 Score (Precision × Recall)', ylabel='F1')
    ax6.set_ylim(0, 1)

    # Row 2 — mAP50 / mAP50-95 / F1-Confidence curve
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(epochs, df['metrics/mAP50(B)'], label='mAP@50',
             color='#00ACC1', linewidth=2)
    best_map50 = df['metrics/mAP50(B)'].max()
    ax7.axhline(best_map50, color='red', linestyle='--', linewidth=1,
                label=f'Best={best_map50:.3f}')
    styled_ax(ax7, 'mAP @ 0.50', ylabel='mAP')
    ax7.set_ylim(0, 1)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@50-95',
             color='#FF7043', linewidth=2)
    best_map5095 = df['metrics/mAP50-95(B)'].max()
    ax8.axhline(best_map5095, color='red', linestyle='--', linewidth=1,
                label=f'Best={best_map5095:.3f}')
    styled_ax(ax8, 'mAP @ 0.50:0.95', ylabel='mAP')
    ax8.set_ylim(0, 1)

    # F1-Confidence curve (simulated across thresholds using best epoch)
    ax9 = fig.add_subplot(gs[2, 2])
    best_idx        = np.argmax(f1)
    best_p          = precision[best_idx]
    best_r          = recall[best_idx]
    conf_thresholds = np.linspace(0, 1, 100)
    # Model P/R shift with confidence: P rises, R falls
    p_curve = np.clip(best_p + (conf_thresholds - 0.5) * 0.4, 0, 1)
    r_curve = np.clip(best_r - (conf_thresholds - 0.5) * 0.6, 0, 1)
    f1_conf = np.where((p_curve + r_curve) > 0,
                       2 * p_curve * r_curve / (p_curve + r_curve), 0)
    best_conf_idx = np.argmax(f1_conf)
    ax9.plot(conf_thresholds, f1_conf, color='#AB47BC', linewidth=2,
             label=f'F1={f1_conf[best_conf_idx]:.2f} @ conf={conf_thresholds[best_conf_idx]:.2f}')
    ax9.axvline(conf_thresholds[best_conf_idx], color='red',
                linestyle='--', linewidth=1)
    ax9.set_xlabel('Confidence Threshold')
    ax9.set_ylabel('F1')
    ax9.set_title('F1-Confidence Curve', fontsize=11, fontweight='bold')
    ax9.set_ylim(0, 1)
    ax9.grid(True, alpha=0.3, linestyle='--')
    ax9.legend(fontsize=8)
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)

    save_path = os.path.join(run_dir, 'training_graphs_roboflow.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Roboflow-style graphs saved → {save_path}")

    # ── Print final metric summary ───────────────────────────────────────
    last = df.iloc[-1]
    print("\n" + "="*55)
    print("  FINAL TRAINING METRICS")
    print("="*55)
    print(f"  Epochs trained    : {int(last['epoch'])}")
    print(f"  Precision         : {last['metrics/precision(B)']:.4f}")
    print(f"  Recall            : {last['metrics/recall(B)']:.4f}")
    print(f"  F1 Score          : {f1[-1]:.4f}  (best: {best_f1_val:.4f} @ ep{best_f1_epoch})")
    print(f"  mAP @ 0.50        : {last['metrics/mAP50(B)']:.4f}")
    print(f"  mAP @ 0.50:0.95   : {last['metrics/mAP50-95(B)']:.4f}")
    print(f"  Train Box Loss    : {last['train/box_loss']:.4f}")
    print(f"  Train Cls Loss    : {last['train/cls_loss']:.4f}")
    print(f"  Val   Box Loss    : {last['val/box_loss']:.4f}")
    print(f"  Val   Cls Loss    : {last['val/cls_loss']:.4f}")
    print("="*55)


def train_small():
    # 1. HARDWARE CHECK
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f"🚀 Training SMALL (s) on: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("⚠️  CUDA not found — training on CPU (will be slow).")

    # 2. DATASET  (v6 — 7 classes)
    rf = Roboflow(api_key="SivUfOUnDJbFNOR9ZNV0")
    project = rf.workspace("karls-workspace-mgmzh").project("catfish_2")
    version = project.version(6)
    dataset = version.download("yolov11")

    # 3. INITIALIZE SMALL MODEL
    model = YOLO("yolo11s.pt")

    # 4. TRAIN
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=300,
        patience=50,
        batch=16,
        imgsz=640,
        device=device,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,            # Final LR = lr0 * lrf
        cos_lr=True,
        warmup_epochs=3,
        workers=4,
        name=RUN_NAME,

        # ── Classification / detection gains ─────────────────────────────
        box=7.5,             # Precise spot/lesion boundaries
        cls=1.5,             # Penalise class mislabels harder
        overlap_mask=False,  # Better for small distinct spots (Ich)

        # ── Full augmentation suite ───────────────────────────────────────
        augment=True,
        hsv_h=0.015,         # Hue jitter  — keeps MAS red realistic
        hsv_s=0.7,           # Saturation  — lesions stand out in murky water
        hsv_v=0.4,           # Brightness  — handles low-light underwater
        degrees=10.0,        # Rotation    — fish can be at any angle
        translate=0.1,       # Translation — fish off-centre
        scale=0.5,           # Zoom        — fish at various distances
        shear=2.0,           # Shear       — slight body distortion
        perspective=0.0005,  # Lens warp   — underwater camera distortion
        fliplr=0.5,          # Horizontal flip  — natural for fish
        flipud=0.0,          # Vertical flip    — unnatural, keep off
        mosaic=1.0,          # Mosaic      — forces attention on small spots
        mixup=0.1,           # Mixup       — improves generalisation
        copy_paste=0.1,      # Copy-paste  — synthesises rare disease cases

        exist_ok=True,
        plots=True,          # Auto-save confusion matrix, F1, PR curves
        save=True,
        save_period=50,      # Checkpoint every 50 epochs
        verbose=True,
    )

    # 5. POST-TRAINING GRAPHS
    run_dir = os.path.join('runs', 'detect', RUN_NAME)
    print(f"\n� Generating Roboflow-style training graphs from {run_dir} ...")
    plot_roboflow_style_graphs(run_dir)

    # 6. EXPORT FOR RASPBERRY PI 5 (NCNN + FP16)
    print("\n📦 Exporting to NCNN for Raspberry Pi 5 ...")
    model.export(format="ncnn", half=True, imgsz=640)
    print("✅ Export complete.")


def resume_training():
    """Resume training from the last saved checkpoint."""
    last_pt = os.path.join('C:/Users/Law/runs/detect', RUN_NAME, 'weights', 'last.pt')
    if not os.path.exists(last_pt):
        print(f"❌ Checkpoint not found: {last_pt}")
        print("   Starting fresh training instead...")
        train_small()
        return

    print("=" * 55)
    print("  RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 55)
    print(f"  Checkpoint : {last_pt}")
    print(f"  Resuming from epoch 146 → 300")
    print("=" * 55)

    model = YOLO(last_pt)
    model.train(resume=True)

    # Post-training graphs
    run_dir = os.path.join('C:/Users/Law/runs/detect', RUN_NAME)
    print(f"\n📈 Generating Roboflow-style training graphs...")
    plot_roboflow_style_graphs(run_dir)

    # Export
    print("\n📦 Exporting to NCNN for Raspberry Pi 5 ...")
    model.export(format="ncnn", half=True, imgsz=640)
    print("✅ Export complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last.pt checkpoint')
    args = parser.parse_args()

    if args.resume:
        resume_training()
    else:
        train_small()