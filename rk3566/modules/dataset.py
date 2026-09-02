"""
GTSRB Dataset
=============
读取 GTSRB (German Traffic Sign Recognition Benchmark) 数据集, 43 类交通标志。

官方数据目录结构:
    训练集: <root>/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/
            <ClassID:05d>/  (43 个子目录, 目录名即类别 0~42)
            每个子目录内为 *.ppm 图片
    测试集: <root>/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/*.ppm
            标签: <root>/GTSRB_Final_Test_GT.csv  (列: Filename, ClassId)

数据增强说明 (重要):
    交通标志的语义对翻转敏感 —— "向左"翻转后变成"向右"。
    因此第一版:
      ✅ 使用: Resize / RandomRotation(小角度) / ColorJitter(轻量)
      ❌ 不使用: RandomHorizontalFlip / RandomVerticalFlip
    (在 README.md 中有详细说明)
"""
import os
import io
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

GTSRB_MEAN = [0.3403, 0.3121, 0.3214]   # GTSRB 常见统计值
GTSRB_STD = [0.2724, 0.2608, 0.2669]

TRAIN_SUBDIR = 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
TEST_IMGDIR = 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'
TEST_CSV = 'GTSRB_Final_Test_GT.csv'


def get_train_transform(input_size: int = 64) -> transforms.Compose:
    """训练集增强: 交通标志旋转是合理增强, 翻转会改变语义故不用。"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(degrees=15),          # 交通标志允许小角度旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
    ])


def get_eval_transform(input_size: int = 64) -> transforms.Compose:
    """验证/测试集: 只做 resize 与归一化, 不做增强。"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
    ])


class GTSRBTrainDataset(Dataset):
    """训练集: 按 <ClassID> 子目录组织, 目录名即标签 (0~42)。"""

    def __init__(self, root: str, transform=None):
        images_dir = os.path.join(root, TRAIN_SUBDIR)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f'GTSRB 训练集目录不存在: {images_dir}\n'
                f'请确认数据集已解压到: {root}')
        self.samples: List[Tuple[str, int]] = []
        for cls_dir in sorted(os.listdir(images_dir)):
            cls_path = os.path.join(images_dir, cls_dir)
            if not os.path.isdir(cls_path):
                continue
            label = int(cls_dir)          # 目录名 00000~00042 -> 0~42
            for fname in sorted(os.listdir(cls_path)):
                if fname.lower().endswith(('.ppm', '.png', '.jpg')):
                    self.samples.append((os.path.join(cls_path, fname), label))
        self.transform = transform
        self.classes = sorted({label for _, label in self.samples})
        assert self.classes == list(range(43)), \
            f'类别不完整: 期望 0~42, 实际 {self.classes}'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class GTSRBTestDataset(Dataset):
    """测试集: 图片在统一目录, 标签从 GT csv 读取。"""

    def __init__(self, root: str, transform=None):
        images_dir = os.path.join(root, TEST_IMGDIR)
        csv_path = os.path.join(root, TEST_CSV)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f'GTSRB 测试集目录不存在: {images_dir}\n'
                f'请确认数据集已解压到: {root}')
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'GTSRB 测试标签文件不存在: {csv_path}')

        # 读取 Filename -> ClassId 映射
        id_map = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)   # 列: Filename, ClassId
            for row in reader:
                id_map[row['Filename']] = int(row['ClassId'])

        self.samples = []
        for fname in sorted(os.listdir(images_dir)):
            if fname.lower().endswith(('.ppm', '.png', '.jpg')):
                if fname not in id_map:
                    raise ValueError(f'测试图片 {fname} 在 GT csv 中无标签')
                self.samples.append((os.path.join(images_dir, fname),
                                     id_map[fname]))
        self.transform = transform
        self.classes = sorted({label for _, label in self.samples})
        assert self.classes == list(range(43)), \
            f'测试集类别不完整: 期望 0~42, 实际 {self.classes}'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def class_distribution(dataset: Dataset) -> List[int]:
    """统计数据集中每个类别的样本数 (用于检查类别分布)。"""
    dist = [0] * 43
    for _, label in dataset:
        dist[label] += 1
    return dist


def stratified_split(dataset: Dataset, val_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    按类别分层划分 train/val 索引 (保证每类在两边都有, 且比例一致)。
    返回 (train_indices, val_indices)。
    """
    import random
    rng = random.Random(seed)
    idx_by_class: dict = {}
    for i, (_, label) in enumerate(dataset):
        idx_by_class.setdefault(label, []).append(i)

    train_idx, val_idx = [], []
    for label in sorted(idx_by_class):
        idxs = idx_by_class[label]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Parquet 数据源 (HuggingFace 镜像 bazyl/GTSRB)
# 官方 zip 下载链接已失效, 使用 HF 的 parquet: 'image'(bytes) + 'label'(int)
# ---------------------------------------------------------------------------
TRAIN_PARQUET = 'train.parquet'
TEST_PARQUET = 'test.parquet'


class GTSRBParquetDataset(Dataset):
    """从 parquet 文件读取 GTSRB, 图片在 bytes 列, 标签在 ClassId 列 (0~42)。

    适配两种 parquet 结构:
      1) image(bytes) + label(int)   —— 常见 HF 结构
      2) Path(dict{'bytes':...}) + ClassId(int) —— bazyl/GTSRB 实际结构
    """

    def __init__(self, parquet_path: str, split: str = 'train',
                 transform=None, max_rows: int = None):
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f'parquet 文件不存在: {parquet_path}')
        df = pd.read_parquet(parquet_path)

        # 识别图片列: 优先 image / Path, 否则取第一个非标签列
        img_col = next((c for c in ('image', 'Path', 'path', 'Image')
                        if c in df.columns), None)
        if img_col is None:
            img_col = [c for c in df.columns if c != 'ClassId'
                       and c != 'label'][0]
        # 识别标签列: 优先 ClassId / label
        lbl_col = next((c for c in ('ClassId', 'label', 'class_id', 'Label')
                        if c in df.columns), 'ClassId')

        if max_rows is not None:
            # 按类别均匀采样 (保证 43 类都覆盖, 避免按顺序截断只剩前几类)
            per_class = max(1, max_rows // 43)
            df = (df.groupby(lbl_col, group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), per_class),
                                              random_state=0))
                    .reset_index(drop=True))

        self.images = df[img_col].tolist()
        self.labels = df[lbl_col].astype(int).tolist()
        self.transform = transform
        self.split = split
        self.classes = sorted(set(self.labels))
        assert self.classes == list(range(43)), \
            f'类别不完整: 期望 0~42, 实际 {self.classes}'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw = self.images[idx]
        # image 列可能是 bytes / dict({'bytes': ...}) / str(base64)
        if isinstance(raw, dict):
            raw = raw.get('bytes', raw.get('path'))
        if isinstance(raw, str):
            import base64
            raw = base64.b64decode(raw)
        img = Image.open(io.BytesIO(bytes(raw))).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def load_gtsrb_dataset(root: str, split: str = 'train', transform=None,
                       max_rows: int = None) -> Dataset:
    """自动选择数据源: parquet 优先, 原始图片目录其次。"""
    train_pq = os.path.join(root, TRAIN_PARQUET)
    if os.path.isfile(train_pq):
        pq = train_pq if split == 'train' else os.path.join(root, TEST_PARQUET)
        return GTSRBParquetDataset(pq, split=split, transform=transform,
                                   max_rows=max_rows)
    if split == 'train':
        return GTSRBTrainDataset(root, transform=transform)
    return GTSRBTestDataset(root, transform=transform)
