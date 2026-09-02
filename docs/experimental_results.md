# Experimental Results（实验结果总览）

统一汇总双平台实验结论。所有数字来自 `results/` 下真实 CSV/JSON。

## 术语约定

| 词 | 含义 |
|----|------|
| MACs | 理论乘加运算量（计算复杂度指标，**≠ 实际执行时间**） |
| measured | 板端实测（warmup+多次取均值） |
| predicted | 模型预测值（已标注时） |
| estimated | 静态估算（未实测，如内存） |

## RK3566（NPU, INT8, 900MHz）

### Latency（22 模型实测 + 12 独立验证）

- benchmark：warmup=5~10 + 50 次，eval_perf 纯 NPU 时间
- **关键发现**：总 MACs 与 latency 相关性低（M1 R²=0.55）；
  算子类型是主要调节因素（FINDING-001）
- **Latency predictor**（M2，实测校准，`rknn_latency_v1.json`）：
  ```
  T(us) = 3.02×Conv_MACs(M) + 980.5×Linear_MACs(M) + 32.0
  验证集 (12 独立模型): R²=0.972, MAE=112.8us, MAPE=9.2%
  ```
- 结论：MACs 与 latency 有关系，但**不能单独解释** NPU 实际执行时间；
  算子类型、tensor shape（flatten≥32768 → FLOAT16 GEMM）影响显著

### Memory（10 模型 eval_memory 实测校准）

- weight：INT8 路径估算误差 <1%（flatten≤16384）
- FLOAT16 GEMM 路径（flatten≥32768）：weight 低估 ~1.85×（fp16 classifier 权重）
- 分段校准后（`rknn_memory_v1.json`）：int8 路径 MAPE 2.6%，fp16 路径 2.3%

## H618（NCNN, Cortex-A53×4, armeabi-v7a）

### 硬件

```
H618: 4×Cortex-A53, Mali-G31, 4GB RAM, Android 31
NCNN 20260526, 输入 64×64×3
```

### CPU vs Vulkan（25 模型实测）

- **CPU 全面优于 Vulkan**（Mali-G31 低端 GPU + 调度开销，Vulkan 慢 1.4~2.5×）
- Vulkan 无 crossover（本 workload 下）
- → 当前 H618 性能建模以 **CPU** 为主要对象
  （注意：不代表 Vulkan 在 H618 上没有价值，仅当前模型规模下无优势）

### CPU latency（25 模型，warmup=20 + 200 次）

- MACs 相关性 0.99（CPU 计算量主导，无 RK3566 的 FLOAT16 GEMM 惩罚）
- **CPU predictor v2**（5-fold CV，`h618_cpu_latency_v2.json`）：
  - M1 (MACs): val R²=0.925, MAPE=8.8%
  - M2 (Conv+Linear): **val R²=0.938, MAPE=6.5%**（选用）

## Cross Hardware（双平台同模型对比）

详见 [cross_hardware_analysis.md](cross_hardware_analysis.md)

| 平台后端 | 相对 RK3566 |
|---------|-------------|
| H618 CPU | 平均慢 ~12×（范围 4~24×） |
| H618 Vulkan | 平均慢 ~28× |

## Flatten/Linear 受控实验（FD4K~FD64K）

详见 `h618/docs/h618_partition_feasibility.md` 与 `h618/docs/figures/`

| 模型 | Flatten | RK3566 (pred) | H618 CPU | RK/H618 |
|------|--------:|--------------:|---------:|--------:|
| FD4K | 4K | 0.59ms | 14.3ms | 24.4× |
| FD16K | 16K | 1.19ms | 17.2ms | 14.5× |
| FD64K | 64K | 3.95ms | 38.7ms | 9.8× |

→ Flatten 增大时双平台差距缩小，但 H618 未形成绝对优势。

## Partition（异构切分可行性）

`T_hetero = T_RK,pre + T_transfer + T_H618,post`

- RK3566 每个算子绝对延迟低于 H618 → 单模型切分无延迟收益（T_hetero > T_RK）
- 切分价值仅在：多模型并发 / 资源受限 / 流水线（未实测验证）
