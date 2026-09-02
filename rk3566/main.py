"""
CLI 入口: 参数化 CNN + GTSRB 训练实验

用法:
    python main.py --mode train   --config configs/model.yaml      # 训练单个模型
    python main.py --mode analyze --config configs/model.yaml      # 仅构建+静态分析
    python main.py --mode search  --config configs/search_space.yaml  # 搜索多个候选

可复现性: 通过 config 中的 seed 固定 random/numpy/torch/CUDA。
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from modules.generator import build_model, describe_model
from modules.analyzer import (analyze_model, format_macs,
                              feature_map_info, count_parameters)
from modules.dataset import (load_gtsrb_dataset, get_train_transform,
                             get_eval_transform, stratified_split)
from modules.trainer import Trainer
from modules.search import (generate_candidates, run_experiment,
                            init_results_file)

# 项目路径 (以 main.py 所在目录为基准)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'GTSRB')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'results.csv')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """检测可用设备。CUDA 检测到但 kernel 不兼容时自动回退 CPU。"""
    if torch.cuda.is_available():
        try:
            # 实测一个小 tensor 操作, 验证 CUDA kernel 与 GPU 架构兼容
            t = torch.zeros(1).cuda()
            t = t + 1
            torch.cuda.synchronize()
            del t
            return 'cuda'
        except Exception as e:
            print(f'  [警告] CUDA 不可用 ({type(e).__name__}: {e}), 回退 CPU')
    return 'cpu'


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'配置文件不存在: {path}')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_data_loaders(data_root: str, input_size: int, batch_size: int,
                      val_ratio: float, seed: int, num_workers: int = 2,
                      max_rows: int = None):
    """加载 GTSRB, 分层划分 train/val, 返回 (train, val, test) DataLoader。"""
    train_ds = load_gtsrb_dataset(data_root, 'train',
                                  get_train_transform(input_size),
                                  max_rows=max_rows)
    test_ds = load_gtsrb_dataset(data_root, 'test',
                                 get_eval_transform(input_size),
                                 max_rows=max_rows)

    train_idx, val_idx = stratified_split(train_ds, val_ratio, seed)
    train_sub = Subset(train_ds, train_idx)
    val_sub = Subset(train_ds, val_idx)
    print(f'  数据: train={len(train_sub)} val={len(val_sub)} '
          f'test={len(test_ds)}')

    # 检查类别分布
    dist = class_distribution_safe(train_sub)
    assert all(c > 0 for c in dist), f'train 集类别不完整: {dist}'

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_sub, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def class_distribution_safe(subset: Subset) -> list:
    dist = [0] * 43
    for i in range(len(subset)):
        _, label = subset[i]
        dist[label] += 1
    return dist


# ---------------------------------------------------------------------------
def mode_analyze(cfg: dict):
    print('===== Analyze Mode =====')
    model_cfg = cfg['model']
    model = build_model(model_cfg)
    print(f'\n模型: {describe_model(model)}')
    print(model)

    stats = analyze_model(model, input_size=model_cfg['input_size'])
    print(f'\n===== 静态分析 (输入 1x3x{model_cfg["input_size"]}'
          f'x{model_cfg["input_size"]}) =====')
    print(f'  Total Parameters : {stats["parameters"]:,}')
    print(f'  Trainable Params : {stats["trainable_parameters"]:,}')
    print(f'  MACs             : {format_macs(stats["macs"])}')
    print(f'  Model Size (FP32): {stats["model_size_MB"]:.2f} MB')
    print('  注: MACs 是理论计算量, 不是硬件推理时延')

    print(f'\n===== 各卷积层 Feature Map =====')
    fm = feature_map_info(model, input_size=model_cfg['input_size'])
    print(f'  {"Layer":<10} {"Input Shape":<18} {"Output Shape":<18} '
          f'{"Channels":<9} Spatial')
    for r in fm:
        print(f'  {r["layer"]:<10} {str(r["input_shape"]):<18} '
              f'{str(r["output_shape"]):<18} {r["channels"]:<9} '
              f'{r["spatial"]}')


# ---------------------------------------------------------------------------
def mode_train(cfg: dict):
    print('===== Train Mode =====')
    model_cfg = cfg['model']
    training_cfg = cfg['training']
    dataset_cfg = cfg.get('dataset', {})
    seed = cfg.get('seed', 42)
    set_seed(seed)
    device = get_device()
    print(f'  设备: {device} | seed: {seed}')

    model = build_model(model_cfg)
    print(f'模型: {describe_model(model)}')
    stats = analyze_model(model, input_size=model_cfg['input_size'])
    print(f'  Parameters: {stats["parameters"]:,} | '
          f'MACs: {format_macs(stats["macs"])} | '
          f'Model Size: {stats["model_size_MB"]:.2f} MB')

    data_root = dataset_cfg.get('root', DATA_ROOT)
    train_loader, val_loader, test_loader = make_data_loaders(
        data_root,
        input_size=model_cfg['input_size'],
        batch_size=training_cfg.get('batch_size', 128),
        val_ratio=dataset_cfg.get('val_ratio', 0.2),
        seed=seed,
        num_workers=training_cfg.get('num_workers', 2),
        max_rows=dataset_cfg.get('max_rows'))

    init_results_file(RESULTS_CSV)
    result = run_experiment(model_cfg, train_loader, val_loader, test_loader,
                            training_cfg, RESULTS_DIR, device, RESULTS_CSV)
    print(f'\n===== 完成: {result["model_id"]} =====')
    print(f'  best_val_acc = {result["best_validation_accuracy"]:.4f}')
    print(f'  test_acc     = {result["test_accuracy"]:.4f}')
    print(f'  结果已记录: {RESULTS_CSV}')


# ---------------------------------------------------------------------------
def mode_search(cfg: dict):
    print('===== Search Mode =====')
    space = cfg['search_space']
    training_cfg = cfg['training']
    dataset_cfg = cfg.get('dataset', {})
    seed = cfg.get('seed', 42)
    set_seed(seed)
    device = get_device()
    print(f'  设备: {device} | seed: {seed}')

    candidates = generate_candidates(
        space,
        num_classes=cfg.get('num_classes', 43),
        input_size=cfg.get('input_size', 64))
    if not candidates:
        print('  没有生成任何候选, 请检查搜索空间')
        return

    data_root = dataset_cfg.get('root', DATA_ROOT)
    train_loader, val_loader, test_loader = make_data_loaders(
        data_root,
        input_size=cfg.get('input_size', 64),
        batch_size=training_cfg.get('batch_size', 128),
        val_ratio=dataset_cfg.get('val_ratio', 0.2),
        seed=seed,
        num_workers=training_cfg.get('num_workers', 2),
        max_rows=dataset_cfg.get('max_rows'))

    init_results_file(RESULTS_CSV)
    for cand in candidates:
        run_experiment(cand, train_loader, val_loader, test_loader,
                       training_cfg, RESULTS_DIR, device, RESULTS_CSV)
    print(f'\n===== Search 完成, 共 {len(candidates)} 个模型 =====')
    print(f'  结果: {RESULTS_CSV}')


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='参数化 CNN + GTSRB 实验')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'analyze', 'search'])
    parser.add_argument('--config', required=True, help='YAML 配置文件路径')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == 'analyze':
        mode_analyze(cfg)
    elif args.mode == 'train':
        mode_train(cfg)
    elif args.mode == 'search':
        mode_search(cfg)


if __name__ == '__main__':
    main()
