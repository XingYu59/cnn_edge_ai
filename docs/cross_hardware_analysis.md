# Cross Hardware Analysis（RK3566 vs H618）

回答：RK3566 和 H618 的性能结构到底有什么不同？

## 1. 结构差异（核心）

双平台 M2 回归系数（T = a·Conv + b·Linear + c，实测拟合）：

| 平台 | conv 开销 | linear 开销 | linear/conv 比值 |
|------|----------|------------|-----------------|
| RK3566 NPU | 3.0 us/M | 975 us/M | **325×** |
| H618 CPU | 89.5 us/M | 1197 us/M | **13×** |

```
RK3566 NPU          H618 CPU
    ↓                   ↓
Conv 非常强          Conv/Linear 成本差异小
Linear 相对成本高     (CPU 通用计算, 无特殊执行路径)
(325x vs conv)
```

**RK3566 的 Linear 惩罚（FLOAT16 GEMM 路径）在 H618 CPU 上不存在**——这是两平台
最本质的结构差异。

## 2. 绝对 vs 相对优势（重要区分）

| 指标 | RK3566 vs H618 |
|------|----------------|
| Conv 绝对 | RK3566 快 ~30× |
| Linear 绝对 | RK3566 快 ~1.2× |
| **Relative** | RK3566 对 linear-heavy 模型优势缩小（但仍在） |
| **Absolute** | RK3566 仍全面更低（当前测试模型） |

> relative advantage ≠ absolute advantage

**H618 的 linear 相对便宜，不代表绝对延迟低**——RK3566 上所有算子仍更快。

## 3. 同模型双平台延迟（实测）

| 平台 | 平均延迟比 (RK3566=1) | 范围 |
|------|----------------------|------|
| H618 CPU | 12.4× | 4.4~23.9× |
| H618 Vulkan | 27.9× | 9.0~57.9× |

- 计算密集模型（B2/V10）：H618 慢 20~24×（RK3566 NPU 优势最大）
- 大 flatten 模型（d3_k3/V11/C1）：H618 只慢 4~8×（RK3566 的 GEMM 惩罚抵消部分优势）

## 4. 对决策的意义

1. 追求单模型最低延迟 → **RK3566 全量**（H618 任何后端都无法超越）
2. 若 H618 必须参与（资源分配/并发）→ 选 **CPU backend**，跑 linear-heavy 相对合适
3. Hardware-aware Search 应优先 RK3566 部署，除非有其他约束

## 5. 图

- `rk3566/docs/figures/rk3566_vs_h618_latency.png`（RK3566 vs H618）
- `h618/docs/figures/h618_cpu_vulkan_analysis.png`（H618 CPU vs Vulkan）
- `h618/docs/figures/h618_flatten_dual_platform.png`（双平台 flatten 行为）
