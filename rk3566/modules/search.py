"""
Search
======
候选模型配置生成 + 批量实验:
- generate_candidates(search_space) : 网格枚举合法配置 (Grid Search)
- validate_search_space             : 检查搜索空间合法性
- run_experiment                    : 单个模型完整实验并写入 results.csv

候选数量: 第一版控制在 10~20 个 (用于验证程序)。
"""
import csv
import os
import time
from typing import Dict, Any, List, Optional

from .generator import build_model, validate_model_config, describe_model
from .analyzer import analyze_model


def validate_search_space(space: Dict[str, Any]) -> None:
    """检查搜索空间结构是否合法。"""
    for key in ('depth', 'kernel_size', 'channels'):
        if key not in space:
            raise ValueError(f'搜索空间缺少字段: {key}')
    if not isinstance(space['depth'], list) or not space['depth']:
        raise ValueError('depth 必须是非空列表')
    if not isinstance(space['kernel_size'], list) or not space['kernel_size']:
        raise ValueError('kernel_size 必须是非空列表')
    if not isinstance(space['channels'], list) or not space['channels']:
        raise ValueError('channels 必须是非空列表')

    for depth in space['depth']:
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f'depth 必须是正整数: {depth}')


def generate_candidates(space: Dict[str, Any],
                        num_classes: int = 43,
                        input_size: int = 64,
                        max_candidates: int = 20,
                        max_classifier_input_dim: int = 20000) -> List[Dict[str, Any]]:
    """
    网格枚举所有 (depth, channels, kernel_size) 合法组合。
    channels 必须与 depth 长度匹配; 不匹配的 depth 组合自动跳过并提示。

    硬件感知过滤 (FINDING-001):
      max_classifier_input_dim 限制 flatten 后 classifier 输入维度。
      实测数据: 16384 正常 (cnn_test 0.91ms), 32768 出现 FLOAT16 Gemm
      瓶颈 (d3_k3 1.59ms)。默认阈值 20000 由 RK3566 实测数据确定。
    """
    from .generator import build_model
    from .analyzer import analyze_model_detail

    validate_search_space(space)
    # 按长度索引 channels 候选
    channels_by_len: Dict[int, List[List[int]]] = {}
    for ch in space['channels']:
        channels_by_len.setdefault(len(ch), []).append(list(ch))

    candidates: List[Dict[str, Any]] = []
    skipped: List[int] = []
    filtered_by_classifier: List[str] = []
    for depth in space['depth']:
        if depth not in channels_by_len:
            skipped.append(depth)
            continue
        for ch in channels_by_len[depth]:
            ch_str = '-'.join(str(c) for c in ch)   # 用通道数序列命名, 保证唯一
            for ks in space['kernel_size']:
                cfg = {
                    'name': f'cnn_d{depth}_c{ch_str}_k{ks}',
                    'depth': depth,
                    'channels': ch,
                    'kernel_size': ks,
                    'num_classes': num_classes,
                    'input_size': input_size,
                }
                # 校验: 非法组合直接抛错 (不静默生成错误模型)
                validate_model_config(cfg)

                # 硬件感知过滤: 检查 classifier 输入维度
                dim = analyze_model_detail(build_model(cfg))['classifier'][
                    'classifier_input_dimension']
                if dim > max_classifier_input_dim:
                    filtered_by_classifier.append(
                        f'{cfg["name"]}({dim})')
                    continue

                candidates.append(cfg)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if skipped:
        print(f'  [search] 跳过 depth={skipped}: '
              f'搜索空间中没有长度匹配的 channels')
    if filtered_by_classifier:
        print(f'  [search] 硬件过滤 {len(filtered_by_classifier)} 个候选 '
              f'(classifier 输入 > {max_classifier_input_dim}): '
              f'{", ".join(filtered_by_classifier)}')
    print(f'  [search] 生成 {len(candidates)} 个候选模型')
    return candidates


def run_experiment(cfg: Dict[str, Any],
                   train_loader, val_loader, test_loader,
                   training_cfg: Dict[str, Any],
                   checkpoint_dir: str,
                   device: str,
                   results_path: str,
                   max_rows_train: Optional[int] = None,
                   ) -> Dict[str, Any]:
    """
    单个候选的完整实验: 构建 -> 静态分析 -> 训练 -> 测试 -> 记录结果。
    返回结果 dict, 同时追加写入 results.csv。
    """
    from .trainer import Trainer

    t0 = time.time()
    print(f'\n===== 实验: {cfg["name"]} =====')
    print(f'  配置: depth={cfg["depth"]} channels={cfg["channels"]} '
          f'kernel={cfg["kernel_size"]}')

    # 1. 构建 + 静态分析
    model = build_model(cfg)
    stats = analyze_model(model, input_size=cfg['input_size'])
    print(f'  Parameters: {stats["parameters"]:,} '
          f'(trainable {stats["trainable_parameters"]:,}) | '
          f'MACs: {stats["macs"]:,} | '
          f'Model Size: {stats["model_size_MB"]:.2f} MB')

    # 2. 训练
    trainer = Trainer(model, training_cfg, device=device)
    best_val_acc = trainer.fit(train_loader, val_loader,
                               checkpoint_dir=checkpoint_dir,
                               model_id=cfg['name'])

    # 3. 测试 (加载 best checkpoint)
    ckpt = os.path.join(checkpoint_dir, f'{cfg["name"]}_best.pt')
    test_metrics = trainer.evaluate(test_loader, checkpoint_path=ckpt)
    print(f'  Test Acc: {test_metrics["test_acc"]:.4f} | '
          f'Test Loss: {test_metrics["test_loss"]:.4f}')

    result = {
        'model_id': cfg['name'],
        'depth': cfg['depth'],
        'channels': str(cfg['channels']),
        'kernel_size': cfg['kernel_size'],
        'input_size': cfg['input_size'],
        'parameters': stats['parameters'],
        'macs': stats['macs'],
        'model_size_MB': stats['model_size_MB'],
        'best_validation_accuracy': round(best_val_acc, 6),
        'test_accuracy': round(test_metrics['test_acc'], 6),
        'training_time': round(trainer.training_time, 2),
    }
    _append_result(results_path, result)
    return result


_RESULT_COLUMNS = ['model_id', 'depth', 'channels', 'kernel_size',
                   'input_size', 'parameters', 'macs', 'model_size_MB',
                   'best_validation_accuracy', 'test_accuracy',
                   'training_time']


def init_results_file(results_path: str) -> None:
    """创建 results.csv 并写入表头 (若不存在)。"""
    if not os.path.isfile(results_path):
        with open(results_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_RESULT_COLUMNS)
            writer.writeheader()
        print(f'  [results] 创建 {results_path}')


def _append_result(results_path: str, result: Dict[str, Any]) -> None:
    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_RESULT_COLUMNS)
        writer.writerow(result)
    print(f'  [results] 已写入 {results_path}: {result["model_id"]}')
