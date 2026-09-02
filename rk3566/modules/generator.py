"""
CNN Generator
=============
根据模型配置自动构建标准 CNN 分类模型（无残差/注意力等复杂结构）。

支持的结构参数（见 configs/model.yaml）:
    depth        : 卷积 block 数量 (3 ~ 8)
    channels     : 每个 block 的输出通道数, 列表长度必须等于 depth
    kernel_size  : 卷积核尺寸 (3 或 5)
    num_classes  : 分类数 (GTSRB = 43)
    input_size   : 输入图像边长 (默认 64)

每个 block: Conv2d -> BatchNorm2d -> ReLU -> (MaxPool2d, 每隔一个 block)

classifier 的输入维度通过 dummy tensor 前向自动推断, 不硬编码。
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any

# 第一版允许的通道数与核尺寸
ALLOWED_CHANNELS = {16, 32, 64, 128}
ALLOWED_KERNELS = {3, 5}
DEFAULT_INPUT_SIZE = 64
DEFAULT_NUM_CLASSES = 43
POOL_EVERY = 2  # 每隔 POOL_EVERY 个 block 做一次 MaxPool2d


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU (same padding)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2  # same conv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ParametricCNN(nn.Module):
    """由 channels / kernel_size / num_classes 参数化的标准 CNN。

    扩展参数 (控制变量实验用):
      pool_positions: 1-based block 索引列表, 指定哪些 block 后池化。
                      None → 使用 pool_every 规则 (向后兼容)。
      kernel_size   : int (全局) 或 list (每层一个 kernel, 长度==depth)。
    """

    def __init__(self, channels: List[int], kernel_size,
                 num_classes: int, pool_every: int = POOL_EVERY,
                 pool_positions: List[int] = None):
        super().__init__()
        self.channels = list(channels)
        if isinstance(kernel_size, int):
            self.kernel_sizes = [kernel_size] * len(self.channels)
        else:
            self.kernel_sizes = list(kernel_size)
        self.num_classes = num_classes
        self.pool_every = pool_every
        self.pool_positions = list(pool_positions) if pool_positions else None

        layers: List[nn.Module] = []
        in_ch = 3  # RGB 输入
        for i, out_ch in enumerate(self.channels):
            layers.append(ConvBlock(in_ch, out_ch, self.kernel_sizes[i]))
            # 池化位置: pool_positions 优先, 否则 pool_every 规则
            if self.pool_positions is not None:
                do_pool = ((i + 1) in self.pool_positions)
            else:
                do_pool = ((i + 1) % pool_every == 0)
            if do_pool:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        # classifier 的输入维度在 build_model() 中通过 shape inference 设置
        self._feature_dim: int = 0
        self.classifier = nn.Linear(self._feature_dim, num_classes)

    def set_feature_dim(self, dim: int):
        """由 shape inference 结果设置 classifier 输入维度。"""
        assert dim > 0, f'invalid feature dim: {dim}'
        self._feature_dim = dim
        self.classifier = nn.Linear(dim, self.num_classes)

    def get_feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def validate_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """校验模型配置, 非法配置直接抛 ValueError。"""
    depth = cfg.get('depth')
    if not isinstance(depth, int) or depth < 1:
        raise ValueError(f'depth 必须是正整数, got: {depth}')
    if not (3 <= depth <= 8):
        raise ValueError(f'第一版 depth 范围 3~8, got: {depth}')

    channels = cfg.get('channels')
    if not isinstance(channels, list) or len(channels) == 0:
        raise ValueError(f'channels 必须是非空列表, got: {channels}')
    if len(channels) != depth:
        raise ValueError(
            f'depth({depth}) 与 channels 长度({len(channels)}) 冲突, '
            f'channels 长度必须等于 depth')
    for c in channels:
        if not isinstance(c, int) or c <= 0:
            raise ValueError(f'channels 元素必须是正整数, got: {c}')
        if c not in ALLOWED_CHANNELS:
            raise ValueError(
                f'第一版 channels 仅允许 {sorted(ALLOWED_CHANNELS)}, got: {c}')

    kernel_size = cfg.get('kernel_size', 3)
    if isinstance(kernel_size, int):
        kernel_list = [kernel_size] * depth
    elif isinstance(kernel_size, list):
        if len(kernel_size) != depth:
            raise ValueError(
                f'kernel_size 列表长度({len(kernel_size)}) 必须等于 depth({depth})')
        kernel_list = list(kernel_size)
    else:
        raise ValueError(f'kernel_size 必须是 int 或 list, got: {kernel_size}')
    for k in kernel_list:
        if k not in ALLOWED_KERNELS:
            raise ValueError(
                f'第一版 kernel_size 仅允许 {sorted(ALLOWED_KERNELS)}, got: {k}')

    pool_positions = cfg.get('pool_positions')
    if pool_positions is not None:
        if not isinstance(pool_positions, list):
            raise ValueError('pool_positions 必须是列表或 None')
        for p in pool_positions:
            if not isinstance(p, int) or not (1 <= p <= depth):
                raise ValueError(
                    f'pool_positions 元素必须是 1~depth 的整数, got: {p}')

    num_classes = cfg.get('num_classes', DEFAULT_NUM_CLASSES)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f'num_classes 必须是 >=2 的整数, got: {num_classes}')

    input_size = cfg.get('input_size', DEFAULT_INPUT_SIZE)
    if not isinstance(input_size, int) or input_size < 8:
        raise ValueError(f'input_size 必须是 >=8 的整数, got: {input_size}')

    return {
        'name': cfg.get('name', 'unnamed'),
        'depth': depth,
        'channels': list(channels),
        'kernel_size': kernel_size,
        'pool_positions': list(pool_positions) if pool_positions else None,
        'num_classes': num_classes,
        'input_size': input_size,
    }


def build_model(cfg: Dict[str, Any]) -> ParametricCNN:
    """根据配置构建 CNN, 并通过 dummy tensor 自动推断 classifier 输入维度。"""
    cfg = validate_model_config(cfg)
    model = ParametricCNN(
        channels=cfg['channels'],
        kernel_size=cfg['kernel_size'],
        num_classes=cfg['num_classes'],
        pool_positions=cfg['pool_positions'],
    )
    # shape inference: 用 dummy tensor 前向 backbone, 得到 flatten 后维度
    with torch.no_grad():
        dummy = torch.randn(1, 3, cfg['input_size'], cfg['input_size'])
        out = model.features(dummy)
    feature_dim = out.numel()
    model.set_feature_dim(feature_dim)
    return model


def describe_model(model: ParametricCNN) -> str:
    """返回模型结构的可读描述, 用于日志/结果记录。"""
    return (f'ParametricCNN(depth={len(model.channels)}, '
            f'channels={model.channels}, kernel={model.kernel_size}, '
            f'num_classes={model.num_classes})')
