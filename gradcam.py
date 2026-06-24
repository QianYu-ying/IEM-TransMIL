import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from TransMIL_main import SliceTransMIL

CSV_FOLDER = r".\results"
MODEL_PATH = r".\results\best_ACC.pth"
OUTPUT_DIR = r".\results\gradcam_results"
NUM_SLICES = 30
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
TARGET_LAYER = "encoder.features"
PROCESS_MODE = "correct_only"   # "correct_only" | "all"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_nifti_and_preprocess(nifti_path, num_slices=NUM_SLICES, target_size=(IMG_SIZE, IMG_SIZE)):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    z = data.shape[2]
    indices = np.linspace(0, z - 1, num_slices, dtype=int)
    slices = []
    for idx in indices:
        slice_2d = data[:, :, idx]
        resized = cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_LINEAR)
        resized = (resized - resized.mean()) / (resized.std() + 1e-6)
        slices.append(resized)
    slices = np.stack(slices, axis=0)[:, np.newaxis, :, :]
    tensor = torch.from_numpy(slices).float().unsqueeze(0)
    return tensor


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.fmap = None
        self.grad = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.fmap = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.grad = grad_out[0].detach()

        target_module = self.model
        for name in self.target_layer.split('.'):
            target_module = getattr(target_module, name)
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        bsz, num_slices = input_tensor.shape[:2]
        logits, _, _, _ = self.model(input_tensor, return_cam=True)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[:, class_idx].backward()

        activations = self.fmap
        gradients = self.grad
        channels, h_feat, w_feat = activations.shape[1], activations.shape[2], activations.shape[3]
        activations = activations.view(bsz, num_slices, channels, h_feat, w_feat)
        gradients = gradients.view(bsz, num_slices, channels, h_feat, w_feat)

        weights = gradients.mean(dim=(3, 4), keepdim=True)
        cam = (weights * activations).sum(dim=2)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.view(bsz * num_slices, 1, h_feat, w_feat),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.view(bsz, num_slices, IMG_SIZE, IMG_SIZE)

        cam_min = cam.view(bsz, num_slices, -1).min(dim=2)[0].view(bsz, num_slices, 1, 1)
        cam_max = cam.view(bsz, num_slices, -1).max(dim=2)[0].view(bsz, num_slices, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(0).cpu().numpy()


def visualize_cam(original_slices, cam_slices, save_path, num_cols=6):
    num_slices = original_slices.shape[0]
    num_rows = (num_slices + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten() if num_slices > 1 else [axes]

    for i in range(num_slices):
        ax = axes[i]
        img = np.rot90(np.fliplr(original_slices[i]), k=1)
        cam = np.rot90(np.fliplr(cam_slices[i]), k=1)
        ax.imshow(img, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f"Slice {i + 1}")
        ax.axis("off")

    for j in range(num_slices, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def select_cases(df):
    if PROCESS_MODE == "all":
        return df
    if PROCESS_MODE == "correct_only":
        return df[df["gt"] == df["pred"]]
    raise ValueError(f"Unsupported PROCESS_MODE: {PROCESS_MODE}")


def main():
    print(f"Using device: {DEVICE}")
    metrics = ["ACC"]

    for metric in metrics:
        print(f"\n========== Processing metric: {metric} ==========")
        model_path = os.path.join(CSV_FOLDER, f"best_{metric}.pth")
        csv_path = os.path.join(CSV_FOLDER, f"test_predictions_{metric}.csv")
        output_subdir = os.path.join(OUTPUT_DIR, metric)
        os.makedirs(output_subdir, exist_ok=True)

        if not os.path.exists(model_path):
            print(f"警告：模型文件不存在 {model_path}，跳过 {metric}")
            continue
        if not os.path.exists(csv_path):
            print(f"警告：CSV文件不存在 {csv_path}，跳过 {metric}")
            continue

        df = pd.read_csv(csv_path)
        process_df = select_cases(df)
        print(f"{metric}: PROCESS_MODE={PROCESS_MODE}, 待处理 {len(process_df)} 个样本")
        if process_df.empty:
            print(f"{metric}: 无可处理样本，跳过")
            continue

        model = SliceTransMIL(n_classes=NUM_CLASSES).to(DEVICE)
        state_dict = torch.load(model_path, map_location="cpu")
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        print(f"{metric}: 模型加载成功")

        gradcam = GradCAM(model, TARGET_LAYER)

        for _, row in tqdm(process_df.iterrows(), total=len(process_df), desc=f"{metric} Grad-CAM"):
            rel_path = str(row["path"])
            if os.path.isabs(rel_path):
                full_path = rel_path
            else:
                full_path = os.path.join(CSV_FOLDER, rel_path)
                if not os.path.exists(full_path):
                    full_path = rel_path
            if not os.path.exists(full_path):
                print(f"  警告：文件不存在 {full_path}，跳过")
                continue

            try:
                input_tensor = load_nifti_and_preprocess(full_path).to(DEVICE)
                original_slices = input_tensor.squeeze(0).squeeze(1).cpu().numpy()
                gt_class = int(row["gt"])
                pred_class = int(row["pred"]) if "pred" in row and pd.notna(row["pred"]) else gt_class
                cam = gradcam.generate(input_tensor, class_idx=pred_class)

                base_name = os.path.basename(full_path).replace(".nii.gz", "").replace(".nii", "")
                status = "correct" if gt_class == pred_class else "wrong"
                save_path = os.path.join(
                    output_subdir,
                    f"{base_name}_gt{gt_class}_pred{pred_class}_{status}_gradcam.png",
                )
                visualize_cam(original_slices, cam, save_path)
            except Exception as e:
                print(f"  处理 {full_path} 时出错: {e}")
                continue


if __name__ == "__main__":
    main()
