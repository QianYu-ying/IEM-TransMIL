import os
import glob
import numpy as np
import nibabel as nib
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
from torchvision.models import *
import torchvision.transforms as T

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder,label_binarize
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import seaborn as sns
import matplotlib.pyplot as plt

from nystrom_attention import NystromAttention

import re
import random
from collections import defaultdict

DATA_ROOT = r"./datasets"
NUM_SLICES = 30
BATCH_SIZE = 32
EPOCHS = 200
PROTO_WARMUP = 0
LR = 1e-4

TARGET_SIZE = (128, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

AUGMENT_TIMES = 1
SAVE_DIR = "./results"
TOPK = 3

ONLY_TEST = False    #  True → val=test
MODE = "predict"       # "train" | "predict"

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def extract_patient_id(path_or_name: str) -> str:
    """
    Extract patient ID from filename.
    Assumes format like 'S0001_left.nii.gz' or 'S0001_right.nii.gz'.
    Returns 'S0001' part.
    """
    name = os.path.basename(path_or_name)
    # Remove extension
    for suf in (".nii.gz", ".npy"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    
    # Split by '_' or '-'
    parts = re.split(r"[_\-]", name)
    
    # Look for pattern like S + 4 digits (e.g., S0001)
    # Or just return the first part if it matches the ID convention
    for p in parts:
        p = p.strip()
        # Matches one letter + 4 digits exactly, or broader like S001
        if re.fullmatch(r"[A-Za-z]\d{4}", p): 
            return p
            
    # Fallback: remove side tokens (left/right) and return the stem
    stem = name
    stem = re.sub(r"(?i)(?:_|-)?(left|right)$", "", stem)
    stem = stem.strip()
    return stem

def split_by_patient_indices(
    samples: list,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Patient-level stratified split.
    Ensures all ears from the same patient go to the same set.
    """
    if len(samples) != len(labels):
        raise ValueError("samples/labels length mismatch")

    n = len(samples)
    num_classes = int(labels.max()) + 1 if n else 0

    # Group indices by patient ID
    groups = defaultdict(list)
    for i, p in enumerate(samples):
        pid = extract_patient_id(p)
        groups[pid].append(i)

    pids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)

    desired_n = int(round(n * val_ratio))
    
    # Calculate desired counts per class in validation set
    total_counts = np.bincount(labels, minlength=num_classes)
    desired_counts = total_counts.astype(np.float64) * val_ratio

    # Pre-calculate stats for each patient
    pid_stats = []
    for pid in pids:
        idxs = groups[pid]
        pid_n = len(idxs)
        pid_counts = np.bincount(labels[idxs], minlength=num_classes)
        pid_stats.append((pid, pid_n, pid_counts))

    # Greedy allocation
    val_pids = set()
    val_n = 0
    val_counts = np.zeros(num_classes, dtype=np.float64)

    # Cost function: prioritize matching class distribution AND total size
    def objective(c_counts, c_n):
        # Size penalty
        size_err = ((c_n - desired_n) / max(1, desired_n)) ** 2
        # Class distribution penalty
        dist_err = 0.0
        if num_classes > 0:
            # Normalized error per class
            errs = (c_counts - desired_counts) / (desired_counts + 1e-6)
            dist_err = float(np.mean(errs ** 2))
        
        return 10.0 * size_err + dist_err

    # Simple greedy pass
    for pid, pid_n, pid_counts in pid_stats:
        current_score = objective(val_counts, val_n)
        new_score = objective(val_counts + pid_counts, val_n + pid_n)
        
        # If adding this patient to val improves (decreases) the deviation from target ratio
        # OR if we are vastly under-filled (early stage)
        if new_score < current_score or val_n < desired_n * 0.5:
            if val_n + pid_n <= desired_n * 1.1: # Don't overshoot too much
                val_pids.add(pid)
                val_counts += pid_counts
                val_n += pid_n

    # Convert to indices
    train_idx, val_idx = [], []
    for pid, idxs in groups.items():
        if pid in val_pids:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return train_idx, val_idx


def load_nii(path):
    return nib.load(path).get_fdata().astype(np.float32)

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def resize_slices(slices, target_size):
    slices = torch.tensor(slices).unsqueeze(1)
    slices = F.interpolate(
        slices, size=target_size,
        mode="bilinear", align_corners=False
    )
    return slices.squeeze(1).numpy()

def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(15),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        T.RandomApply([T.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.3)
    ])


class NiiSliceMILDataset(Dataset):
    def __init__(self, root_dir, split,
                 num_slices, transform=None, augment_times=1,small_ids=None):

        self.num_slices = num_slices
        self.transform = transform

        split_dir = os.path.join(root_dir, split)
        self.class_names = sorted(os.listdir(split_dir))
        self.le = LabelEncoder()
        self.le.fit(self.class_names)

        base_samples, base_labels = [], []
        for cls in self.class_names:
            for f in glob.glob(os.path.join(split_dir, cls, "*.nii.gz")):
                base_samples.append(f)
                base_labels.append(cls)

        base_labels = self.le.transform(base_labels)

        self.samples, self.labels = [], []
        
        for path, lab in zip(base_samples, base_labels):
            if small_ids is not None and lab in small_ids:
                repeat = augment_times
            else:
                repeat = 1
            for _ in range(repeat):
                self.samples.append(path)
                self.labels.append(lab)
                
        self.base_len = len(base_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol = load_nii(self.samples[idx])
        label = self.labels[idx]

        D = vol.shape[2]
        idxs = np.linspace(0, D - 1, self.num_slices).astype(int)
        slices = vol[:, :, idxs]
        slices = np.transpose(slices, (2, 0, 1))
        slices = resize_slices(slices, TARGET_SIZE)
        slices = normalize(slices)

        slices = torch.tensor(slices).unsqueeze(1)

        if self.transform and idx >= self.base_len:
            slices = torch.stack([self.transform(s) for s in slices])

        return slices, torch.tensor(label).long(), self.samples[idx]

class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=True):
        super().__init__()

        assert name in ["resnet18", "resnet34", "resnet50"]

        net = getattr(models, name)(pretrained=pretrained)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        
        if name in ["resnet18", "resnet34"]:
            self.out_channels = [128, 256, 512]
        else:  # resnet50
            self.out_channels = [512, 1024, 2048]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f2, f3, f4]
    
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k, rope):
    cos = rope.cos()[None, None, :, :]
    sin = rope.sin()[None, None, :, :]

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.relative_bias_table = nn.Parameter(
            torch.zeros(2 * max_len - 1, num_heads)
        )

        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self, seq_len):
        """
        seq_len: number of tokens (including CLS)
        """
        coords = torch.arange(seq_len)
        relative_coords = coords[None, :] - coords[:, None]

        relative_coords = relative_coords.clamp(
            -self.max_len + 1,
            self.max_len - 1
        )

        relative_coords += self.max_len - 1

        bias = self.relative_bias_table[relative_coords]
        bias = bias.permute(2, 0, 1)
        return bias


class AttentionWithRelBiasRoPE(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64, max_len=100):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.rope = RotaryEmbedding(dim_head)
        self.relative_bias = RelativePositionBias(
            num_heads=heads,
            max_len=max_len
        )

    def forward(self, x,return_attn=False):
        B, N, C = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, N, self.heads, -1)
                         .transpose(1, 2),
            qkv
        )
        rope = self.rope(N, x.device)
        q, k = apply_rope(q, k, rope)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.relative_bias(N)
        attn = attn + bias.unsqueeze(0)

        attn_softmax = attn.softmax(dim=-1)

        out = attn_softmax @ v
        out = out.transpose(1, 2).reshape(B, N, -1)
        if return_attn:
            return self.to_out(out), attn_softmax
        return self.to_out(out)

# class PatchEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = resnet50(pretrained=True)
#         backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
#         self.encoder = nn.Sequential(*list(backbone.children())[:-1])
#     def forward(self, x):
#         return self.encoder(x).flatten(1)

class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x, return_map=False):
        fmap = self.features(x)
        feat = self.pool(fmap).flatten(1)

        if return_map:
            return feat, fmap
        return feat

class PositionalEncoding3D(nn.Module):
    def __init__(self, dim, num_slices):
        super().__init__()
        self.slice_embedding = nn.Embedding(num_slices, dim)
    def forward(self, x):
        B, K, D = x.shape
        positions = torch.arange(K, device=x.device).unsqueeze(0)
        pos_emb = self.slice_embedding(positions)
        return x + pos_emb

class TransLayer(nn.Module):
    def __init__(self, dim=512,max_len=100,layer_name='RoPE'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if layer_name == 'NysA':
            self.attn = NystromAttention(
                dim=dim, heads=8, dim_head=64,
                num_landmarks=dim // 2, residual=True,dropout=0.1
            )
        elif layer_name == 'RoPE':
            self.attn = AttentionWithRelBiasRoPE(
                            dim=dim,
                            heads=8,
                            dim_head=64,
                            max_len=max_len
                        )
    def forward(self, x,return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(self.norm(x), return_attn=True)
            return x + attn_out, attn
        else:
            return x + self.attn(self.norm(x))
    
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        cls, feat = x[:, :1], x[:, 1:]
        B, _, C = feat.shape
        feat = feat.transpose(1, 2).view(B, C, H, W)
        feat = feat + self.proj(feat) + self.proj1(feat) + self.proj2(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return torch.cat([cls, feat], dim=1)


class TransMIL(nn.Module):
    def __init__(self, n_classes,max_len=100,layer_name='NysA'):
        super(TransMIL, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(512,max_len,layer_name)
        self.layer2 = TransLayer(512,max_len,layer_name)
        self.pos = PPEG(512)
        self.norm = nn.LayerNorm(512)
        self.head = nn.Linear(512, n_classes)
    def forward(self, feats,return_attn=False):
        h = self.fc1(feats)
        B, N, _ = h.shape
        H = W = int(np.ceil(np.sqrt(N)))
        pad = H * W - N
        if pad > 0:
            h = torch.cat([h, h[:, :pad]], dim=1)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.layer1(h)
        h = self.pos(h, H, W)
        if return_attn:
            h, attn = self.layer2(h, return_attn=True)   # 获取第二层的注意力
        else:
            h = self.layer2(h)

        feat = self.norm(h)[:, 0]
        logits = self.head(feat)
        if return_attn:
            return logits, feat, attn
        return logits, feat


class SliceTransMIL(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = PatchEncoder()
        self.pos = PositionalEncoding3D(2048, NUM_SLICES)
        self.transmil = TransMIL(n_classes,max_len=100,layer_name='RoPE') # max_len=100
        
    def forward(self, x, return_cam=False, return_attn=False):
        B, K, _, H, W = x.shape
        x = x.view(B*K, 1, H, W)

        if return_cam:
            feats, fmap = self.encoder(x, return_map=True)
        else:
            feats = self.encoder(x)

        feats = feats.view(B, K, -1)
        feats = self.pos(feats)

        if return_attn:
            logits, bag_feat, attn = self.transmil(feats, return_attn=True)
        else:
            logits, bag_feat = self.transmil(feats)

        # 根据返回标志组合输出
        if return_cam and return_attn:
            return logits, bag_feat, feats, fmap, attn
        elif return_cam:
            return logits, bag_feat, feats, fmap
        elif return_attn:
            return logits, bag_feat, feats, attn
        else:
            return logits, bag_feat, feats
    

class SliceLoss(nn.Module):
    def __init__(self, alpha=None,small_ids=None,gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.small_ids = small_ids
        self.ce = nn.CrossEntropyLoss(weight=alpha)
        self.weights = nn.Parameter(torch.tensor([1.0, 0.3, 0.2]))
    
    def prototype_loss(self, feat, label):

        mask = torch.zeros_like(label, dtype=torch.bool)
        for c in self.small_ids:
            mask |= (label == c)

        if mask.sum() < 2: 
            return torch.tensor(0.0, device=feat.device)

        loss = 0
        for c in torch.unique(label[mask]):
            f = feat[(label == c)]
            proto = f.mean(0, keepdim=True)
            loss += ((f - proto) ** 2).mean()

        return loss / len(torch.unique(label[mask]))

    def forward(self, logits,feat,target,use_proto=False):
        w_ce, w_focal, w_proto = self.weights
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            logpt = logpt * self.alpha[target]
        
        ce_loss = self.ce(logits,target)
        focal_loss = (-((1 - pt) ** self.gamma) * logpt).mean()
        proto_loss = self.prototype_loss(feat,target) 
        
        total_loss = w_ce * ce_loss + w_focal * focal_loss + w_proto * proto_loss if use_proto else w_ce * ce_loss + w_focal * focal_loss

        return total_loss
    
# class EMA:
#     def __init__(self, model, decay):
#         self.decay = decay
#         self.shadow = {}
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()

#     def update(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = (
#                     self.decay * self.shadow[name]
#                     + (1.0 - self.decay) * param.data
#                 )

#     def apply_shadow(self, model):
#         self.backup = {}
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.backup[name] = param.data.clone()
#                 param.data = self.shadow[name]

#     def restore(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 param.data = self.backup[name]


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion,
                 small_ids, device, save_dir,
                 topk, only_test):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.small_ids = small_ids
        self.device = device
        self.save_dir = save_dir
        self.topk = topk
        self.only_test = only_test
        self.best_scores = {
                                "ACC": 0,
                                "Recall": 0,
                                "F1_macro": 0,
                                "Precision": 0
                            }
        os.makedirs(self.save_dir, exist_ok=True)

    # ==========================================================
    # Data
    # ==========================================================
    def build_loaders(self):

        full = NiiSliceMILDataset(
            DATA_ROOT, "train",
            NUM_SLICES, get_train_transform(),
            AUGMENT_TIMES, self.small_ids
        )
        
        if self.only_test:
            train_set = full
            val_set = NiiSliceMILDataset(
                DATA_ROOT, "test", NUM_SLICES
            )
        else:
            base_full = NiiSliceMILDataset(
                DATA_ROOT, "train",
                NUM_SLICES,
                transform=None,
                augment_times=1,
                small_ids=None
            )

            labels = np.array(base_full.labels)
            samples = list(base_full.samples)
            
            # Use patient-level split instead of simple random shuffle
            train_base_idx, val_base_idx = split_by_patient_indices(
                samples=samples,
                labels=labels,
                val_ratio=0.2,
                seed=SEED
            )

            # Reconstruct training set with augmentation support
            # We need to map base indices back to augmented indices
            train_full = NiiSliceMILDataset(
                DATA_ROOT, "train",
                NUM_SLICES,
                transform=get_train_transform(),
                augment_times=AUGMENT_TIMES,
                small_ids=self.small_ids
            )

            # Map base sample path -> all augmented indices in train_full
            # Because NiiSliceMILDataset with augment_times > 1 repeats samples
            path_to_indices = defaultdict(list)
            for j, p in enumerate(train_full.samples):
                path_to_indices[p].append(j)

            train_indices = []
            for i in train_base_idx:
                path = base_full.samples[i]
                train_indices.extend(path_to_indices[path])
            
            train_set = Subset(train_full, train_indices)
            val_set = Subset(base_full, val_base_idx)
            
            # Print split statistics
            print(f"Split Summary (Patient-level):")
            print(f"  Train samples (base): {len(train_base_idx)}")
            print(f"  Val samples: {len(val_base_idx)}")
            print(f"  Train patients (approx): {len(set([extract_patient_id(samples[i]) for i in train_base_idx]))}")
            print(f"  Val patients (approx): {len(set([extract_patient_id(samples[i]) for i in val_base_idx]))}")
            
            # labels = np.array(full.labels)
            # train_idx, val_idx = [], []

            # for c in np.unique(labels):
            #     idx = np.where(labels == c)[0]
            #     np.random.shuffle(idx)
            #     split = int(0.8 * len(idx))
            #     train_idx += idx[:split].tolist()
            #     val_idx += idx[split:].tolist()

            # train_set = Subset(full, train_idx)
            # val_set = Subset(full, val_idx)

        self.train_loader = DataLoader(
            train_set, BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_set, BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    # ==========================================================
    # Train
    # ==========================================================
    def train(self, class_names):

        self.build_loaders()

        for epoch in range(EPOCHS):

            self.model.train()
            total_loss = 0

            train_preds, train_gts, train_probs = [], [], []

            train_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS}"
            )

            for x, y, _ in train_bar:

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=True):
                    logits, feat, _ = self.model(x)
                    prob = torch.softmax(logits, dim=1)

                    if epoch >= PROTO_WARMUP:
                        loss = self.criterion(
                            logits, feat, y, use_proto=True
                        )
                    else:
                        loss = self.criterion(
                            logits, feat, y
                        )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                train_preds.extend(
                    logits.argmax(1).detach().cpu().numpy()
                )
                train_gts.extend(
                    y.detach().cpu().numpy()
                )
                train_probs.extend(
                    prob.detach().cpu().numpy()
                )

            train_metrics = self.compute_metrics(
                np.array(train_gts),
                np.array(train_preds),
                np.array(train_probs),
                class_names,
                stage="train"
            )

            val_metrics = self.validate(class_names)

            print(f"\nEpoch {epoch+1}")
            print("Train:", train_metrics)
            print("Val:", val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()
                
    def get_topk_slices_for_sample(self, x, k=None):
        """
        输入 x: 一个病例的切片数据，形状 (K, 1, H, W) 或 (1, K, 1, H, W)
        返回 (indices, scores)
        """
        if k is None:
            k = self.topk

        self.model.eval()
        with torch.no_grad():
            if x.dim() == 4:          # (K, 1, H, W)
                x = x.unsqueeze(0)    # -> (1, K, 1, H, W)

            logits, bag_feat, feats, attn = self.model(x, return_attn=True)

        # attn 形状: (B, heads, N, N)，B=1
        # 取 [CLS] token（索引0）对所有切片（索引1到N-1）的注意力
        cls_attn = attn[0, :, 0, 1:]          # (heads, K)
        scores = cls_attn.mean(dim=0)          # (K,)
        topk_scores, topk_indices = torch.topk(scores, k)

        # 返回 0-based 内部索引, 1-based 可视化索引, 以及分数
        topk_indices_1based = (topk_indices + 1).cpu().tolist()
        return topk_indices.cpu().tolist(), topk_indices_1based, topk_scores.cpu().tolist()
    # ==========================================================
    # Evaluate
    # ==========================================================                
    def evaluate(self, loader, stage, class_names,
             metric_for_saving=True,collect_topk=True):
        
        self.model.eval()

        preds, gts, probs, paths_all = [], [], [], []
        topk_indices_all = []  
        topk_scores_all = []

        # 创建保存 topk 图片的文件夹
        if collect_topk and "test" in stage:
            topk_img_dir = os.path.join(self.save_dir, f"{stage}_topk_images")
            os.makedirs(topk_img_dir, exist_ok=True)

        with torch.no_grad():
            bar = tqdm(loader, desc=f"{stage} Evaluating")

            for x, y, path in bar:

                x = x.to(self.device)

                if collect_topk:
                    batch_topk_indices = []
                    batch_topk_scores = []
                    for i in range(x.size(0)):
                        indices0, indices1, scores = self.get_topk_slices_for_sample(x[i])
                        batch_topk_indices.append(indices1)
                        batch_topk_scores.append(scores)

                        # 保存 Top-K 图片逻辑
                        if "test" in stage:
                            case_name = os.path.basename(path[i]).split('.')[0]
                            for rank, slice_idx0 in enumerate(indices0):
                                slice_idx1 = indices1[rank]
                                score = scores[rank]

                                # 提取对应切片并转为 numpy（使用0-based内部索引）
                                img = x[i, slice_idx0, 0].cpu().numpy()
                                
                                # 为了迎合临床阅片习惯（翻转和旋转）
                                img = np.fliplr(img)
                                img = np.rot90(img, k=1)

                                # 简单的归一化显示 (0-255)
                                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                                
                                img_save_name = (
                                    f"{case_name}_rank{rank+1}_sliceIdx{slice_idx1}"
                                    f"_score{score:.3f}.png"
                                )
                                plt.imsave(os.path.join(topk_img_dir, img_save_name), img, cmap='gray')

                    logits, _, _, _ = self.model(x, return_attn=True)
                else:
                    logits, _, _ = self.model(x)
                prob = torch.softmax(logits, dim=1)

                preds.extend(
                    logits.argmax(1).cpu().numpy()
                )
                gts.extend(
                    y.numpy()
                )
                probs.extend(
                    prob.cpu().numpy()
                )
                paths_all.extend(path)
                
                if collect_topk:
                    topk_indices_all.extend(batch_topk_indices)
                    topk_scores_all.extend(batch_topk_scores)

        gts = np.array(gts)
        preds = np.array(preds)
        probs = np.array(probs)

        metrics = self.compute_metrics(
            gts, preds, probs,
            class_names,
            stage=stage
        )
        self.save_predictions(
            paths_all,gts,preds,
            probs,stage
        )
        if collect_topk:
            df_topk = pd.DataFrame({
                "path": paths_all,
                "topk_indices_1based": [str(idx) for idx in topk_indices_all],
                "topk_scores": [str(score) for score in topk_scores_all]
            })
            df_topk.to_csv(os.path.join(self.save_dir, f"{stage}_topk.csv"), index=False)
        if metric_for_saving is not None:
            self.save_metric_visual(
                gts, preds, probs,
                class_names,
                metric_for_saving,
                metrics[metric_for_saving],
                stage
            )

        return metrics, gts, preds, probs, paths_all


    # ==========================================================
    # Validate
    # ==========================================================
    def validate(self, class_names):

        metrics, gts, preds, probs, _ = self.evaluate(
            self.val_loader,
            stage="val",
            class_names=class_names
        )

        for metric_name in self.best_scores.keys():

            current_value = metrics[metric_name]

            if current_value > self.best_scores[metric_name]:

                self.best_scores[metric_name] = current_value

                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_dir,
                        f"best_{metric_name}.pth"
                    )
                )

                self.save_metric_visual(
                    gts, preds, probs,
                    class_names,
                    metric_name,
                    current_value,
                    stage="val"
                )

        return metrics
    
    # ==========================================================
    # Predict
    # ==========================================================
    def predict(self, class_names, metric_name="ACC",collect_topk=True):

        print("Starting prediction for", metric_name)

        model_path = os.path.join(
            self.save_dir,
            f"best_{metric_name}.pth"
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)

        test_set = NiiSliceMILDataset(
            DATA_ROOT, "test", NUM_SLICES
        )
        test_loader = DataLoader(
            test_set,
            BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        metrics, gts, preds, probs, paths = self.evaluate(
            test_loader,
            stage=f"test_{metric_name}",
            class_names=class_names,
            metric_for_saving=metric_name,
            collect_topk=True
        )

        df = pd.DataFrame({
            "path": paths,
            "gt": gts,
            "pred": preds
        })

        df.to_csv(
            os.path.join(
                self.save_dir,
                f"test_predictions_{metric_name}.csv"
            ),
            index=False
        )

        print("Test Metrics:", metrics)

    # ==========================================================
    # Metrics
    # ==========================================================
    def compute_metrics(self,
                        gts, preds, probs,
                        class_names,
                        stage):

        acc = accuracy_score(gts, preds)
        precision = precision_score(gts, preds, average="macro", zero_division=0)
        recall = recall_score(gts, preds, average="macro", zero_division=0)
        f1_macro = f1_score(gts, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(gts, preds, average="weighted")

        cm = confusion_matrix(gts, preds)

        # specificity
        specificity_list = []
        for i in range(len(cm)):
            tn = np.sum(cm) - (
                np.sum(cm[i, :]) +
                np.sum(cm[:, i]) -
                cm[i, i]
            )
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity_list.append(
                tn / (tn + fp + 1e-8)
            )
        specificity = np.mean(specificity_list)

        # AUC
        try:
            gts_bin = label_binarize(
                gts,
                classes=np.unique(gts)
            )
            auc = roc_auc_score(
                gts_bin,
                probs,
                average="macro",
                multi_class="ovr"
            )
        except:
            auc = 0

        metrics_dict = {
            "ACC": acc,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1_macro": f1_macro,
            "F1_weighted": f1_weighted,
        }
        
        csv_path = os.path.join(
            self.save_dir,
            f"{stage}_metrics.csv"
        )

        df = pd.DataFrame([metrics_dict])

        if os.path.exists(csv_path):
            df.to_csv(
                csv_path,
                mode="a",
                header=False,
                index=False
            )
        else:
            df.to_csv(
                csv_path,
                index=False
            )

        return metrics_dict
    
    # ==========================================================
    # Best Visualizations
    # ==========================================================
    def save_metric_visual(self,
                        gts, preds, probs,
                        class_names,
                        metric_name,
                        metric_value,
                        stage):

        # 注释掉混淆矩阵绘制
        # cm = confusion_matrix(
        #     gts, preds,
        #     normalize="true"
        # )
        # plt.figure(figsize=(6, 5))
        # disp = ConfusionMatrixDisplay(
        #     confusion_matrix=cm,
        #     display_labels=class_names
        # )
        # disp.plot(cmap="Blues", values_format=".2f")
        # plt.title(
        #     f"{stage.upper()} - Best {metric_name}\n"
        #     f"{metric_name} = {metric_value:.4f}"
        # )
        # plt.savefig(
        #     os.path.join(
        #         self.save_dir,
        #         f"{stage}_best_{metric_name}_confusion_matrix.png"
        #     ),
        #     dpi=300,
        #     bbox_inches="tight"
        # )
        # plt.close()

        try:
            gts_bin = label_binarize(
                gts,
                classes=np.unique(gts)
            )

            plt.figure(figsize=(6, 5))

            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(
                    gts_bin[:, i],
                    probs[:, i]
                )
                plt.plot(
                    fpr, tpr,
                    label=f"{class_names[i]}"
                )

            plt.plot([0, 1], [0, 1], "--")

            plt.title(
                f"{stage.upper()} - ROC\n"
                f"Best {metric_name} = {metric_value:.4f}"
            )

            plt.legend()

            plt.savefig(
                os.path.join(
                    self.save_dir,
                    f"{stage}_best_{metric_name}_roc.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        except:
            pass
        
    def save_predictions(self,paths,gts,preds,probs,stage):
    
        df_dict = {
            "path": paths,
            "gt": gts,
            "pred": preds
        }

        for i in range(probs.shape[1]):
            df_dict[f"prob_class_{i}"] = probs[:, i]

        df = pd.DataFrame(df_dict)

        save_path = os.path.join(
            self.save_dir,
            f"{stage}_predictions.csv"
        )

        df.to_csv(save_path, index=False)
        
    def get_topk_slices(self, x):
        self.eval()
        with torch.no_grad():
            logits, bag_feat, feats, attn = self.forward(x, return_attn=True)

        # attn 形状: (B, heads, N, N)，其中 N = 切片数 + 1 (cls)
        # 取 [CLS] token (索引 0) 对所有切片 (索引 1 到 N-1) 的注意力
        cls_attn = attn[0, :, 0, 1:]               # 形状 (heads, K)
        # 对所有头取平均，得到每个切片的重要性分数
        scores = cls_attn.mean(dim=0)               # 形状 (K,)

        # 获取分数最高的 k 个切片的索引（降序）
        topk_scores, topk_indices = torch.topk(scores, self.topk)
        return topk_indices.cpu().tolist(), topk_scores.cpu().tolist()


def main():
    dummy = NiiSliceMILDataset(DATA_ROOT, "train", NUM_SLICES)
    model = SliceTransMIL(len(dummy.class_names)).to(DEVICE)
    
    counts = Counter(dummy.labels[:dummy.base_len])
    small_ids = [k for k,v in counts.items() if v < 500]
    alpha = torch.tensor(
        [1.0 / counts.get(i,1) for i in range(len(dummy.class_names))],
        device=DEVICE
    )
    alpha /= alpha.sum()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = SliceLoss(alpha,small_ids) # nn.CrossEntropyLoss(weight=alpha)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=15,T_mult=2,eta_min=1e-4)
    trainer = Trainer(
        model, optimizer,scheduler, criterion,small_ids,
        DEVICE, SAVE_DIR, TOPK, ONLY_TEST
    )

    if MODE == "train":
        trainer.train(dummy.class_names)
    else:
        trainer.predict(dummy.class_names, metric_name="ACC")

if __name__ == "__main__":
    main()
