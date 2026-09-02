# H618 NCNN 性能建模报告

**日期**：2026-09-02 | **平台**：H618 (Android 31, armeabi-v7a, Cortex-A53×4, Mali-G31)
**Runtime**：NCNN 20260526 (Vulkan 预编译) | **输入**：GTSRB 64×64×3

---

## 1. 链路与 Benchmark 方法

```
PyTorch CNN → ONNX → pnnx → NCNN param/bin → h618_ncnn_bench → H618
```

- 模型：14 个（与 RK3566 完全相同的结构，保证跨平台可比）
- 方法：warmup=20 + iterations=200，取 mean/median/min/max/std/p95
- Backend：CPU (4 线程) 与 Vulkan (Mali-G31)
- 数据：`results/h618_latency.csv` / `results/h618_dataset.csv`

## 2. 关键结果

### 2.1 CPU 与 Vulkan 延迟

| 模型 | MACs | CPU (ms) | Vulkan (ms) | CPU/Vulkan |
|------|-----:|---------:|------------:|-----------:|
| d3_k3 | 31.5M | 7.14 | 20.19 | 0.354 |
| cnn_test | 78.0M | 10.34 | 24.04 | 0.430 |
| d5_k3 | 118M | 15.83 | 29.12 | 0.544 |
| d5_k5 | 326M | 35.17 | 89.48 | 0.393 |
| C1 (flatten 65536) | 271M | 30.99 | 44.81 | 0.692 |
| ... | | | | |

**所有 14 个模型 Vulkan 均慢于 CPU**（ratio 0.34~0.71）。

### 2.2 相关性（Pearson, vs latency）

| 指标 | CPU | Vulkan |
|------|----:|-------:|
| **MACs** | **0.987** | **0.918** |
| Params | 0.451 | 0.153 |
| Conv MACs | 0.986 | 0.920 |
| Linear MACs / Flatten | 0.330 | 0.020 |

**H618 CPU 上 MACs 高度解释延迟**（0.987）——与 RK3566 NPU 截然不同
（RK3566 总 MACs 仅 0.74，需分算子）。CPU 是通用计算单元，无 FLOAT16 GEMM 惩罚。

### 2.3 Predictor（holdout: 10 train / 4 val）

| Backend | 模型 | Train R² | Val R² | Val MAPE |
|---------|------|---------|--------|----------|
| CPU | T=a·Conv+b·Lin+c | 0.992 | 0.985 | **7.4%** |
| Vulkan | T=a·MACs+b | 0.868 | 0.804 | 27.9% |

- CPU predictor 可靠（MAPE 7.4%）；Vulkan 预测差（Mali-G31 调度噪声大）
- 模型文件：`models/h618_cpu_latency_v1.json` / `h618_vulkan_latency_v1.json`

### 2.4 RK3566 vs H618（相同模型）

| | 平均 | 范围 |
|--|-----|------|
| H618 CPU / RK3566 NPU | **12.4×** | 4.4~23.9× |
| H618 Vulkan / RK3566 NPU | **27.9×** | 9.0~57.9× |

RK3566 NPU 全面优势。**关键差异**：RK3566 的 FLOAT16 GEMM 惩罚
（大 flatten 模型）在 H618 CPU 上**不存在**——d3_k3 CPU 只慢 4.4×、
V11 只慢 6.1×，而计算密集模型（V10/B2）慢 20~24×。

## 3. 回答研究问题

### Q1: MACs 与 H618 latency 是否高度相关？
**CPU 上高度相关**（Pearson 0.987, M1 拟合 R²=0.970）；Vulkan 上中等（0.918）。

### Q2: CPU 与 Vulkan 对结构敏感性不同？
**是**。CPU 完全由计算量（MACs）主导；Vulkan 上 MACs 解释力下降
（0.918 vs 0.987），小模型调度开销占比更大（Vulkan 固定开销 ~15-20ms 起）。

### Q3: 哪些 operator 贡献最大？
Conv 主导（CPU）。Linear/GEMM 在 H618 CPU 上无特殊惩罚
（Linear MACs 相关性仅 0.33，因其占比小且 CPU 无 fp16 问题）。

### Q4: 小模型是否有 Vulkan 开销？
**是，非常明显**。最小模型 A1（6.7ms CPU）Vulkan 也要 19.6ms
（~13ms 固定开销）；Vulkan 对小模型（<50M MACs）无优势。

### Q5: 是否存在 CPU→Vulkan crossover？
**未观察到**。全部 14 个模型（6.7~35ms CPU 范围）Vulkan 均慢。
Mali-G31 低端 GPU + NCNN Vulkan 在本规模下无 crossover。

### Q6: RK3566 与 H618 对不同模型的优势是否不同？
**是**：
- RK3566 对**计算密集**模型优势最大（V10: H618 CPU 慢 24×）
- RK3566 对**大 flatten** 模型优势相对小（V11: 慢 6.1×，因 NPU 上 FLOAT16 GEMM 惩罚）
- 即：**RK3566 适合计算密集/深层模型；若必须用 H618，CPU backend 优于 Vulkan**

## 4. 产出清单

```
cnn/h618/
├── ncnn_bench/                 # benchmark 工程 (CPU/Vulkan)
├── convert_to_ncnn.py          # PyTorch→ONNX→pnnx→ncnn
├── models/                     # 14 模型 param/bin + predictor json
├── results/h618_latency.csv    # 原始 28 行 (14×2)
├── results/h618_dataset.csv    # 合并静态特征
├── docs/figures/               # 分析图
└── (分析/拟合/对比/验证脚本)
```

## 5. Limitations

- 14 模型样本有限，predictor 为第一版（holdout 验证）
- Vulkan 延迟噪声大（std 高），predictor 误差大
- 固定 4 CPU 线程、单输入尺寸（64×64）
- 未测 fp16/int8 优化选项（NCNN 默认 fp32）
- Mali-G31 是低端 GPU，结论不代表所有 Vulkan 设备

## 6. 下一步

1. Hardware-aware Search 的三平台输入已齐（RK3566 predictor + H618 CPU predictor）
2. 如需提升 Vulkan 预测：增加样本 + 检查 NCNN Vulkan 算子配置
3. 异构任务分配决策：计算密集 → RK3566；实时性要求高时用 RK3566 NPU
