# RK3566 Memory Analysis 报告

**内存分析与 Memory Estimator 校准**（第一版）
**日期**：2026-09-01 | **平台**：K11C / RK3566, librknnrt 1.5.2

---

## 1. Memory 由哪些部分组成？

```
M_total ≈ M_weight + M_activation + M_runtime_overhead
  M_weight    = params × bytes_per_param   (INT8=1 / FP16=2 / FP32=4)
  M_activation= Σ C×H×W (每层特征图), 峰值取最大单层
  M_overhead  = internal tensor buffers / workspace / alignment (实测可见)
```

RKNN `eval_memory()` 实测提供：`weight_memory` / `internal_memory` / `total_memory`。

## 2. Parameters 如何影响 Weight Memory？

**定义恒等**：`M_weight = params × bytes`。INT8 路径下 weight 估算与实测**误差 <1%**（cnn_test/A5/B2/C3 均 ~1.0 倍）。

## 3. Feature Map 如何影响 Activation Memory？

每层 `C×H×W×bytes`，峰值来自**第一层 conv 输出**（64×64 分辨率）：
- 32 通道 → 131,072B；64 通道 → 262,144B
- 实测 internal_memory（208~536KB）**高于单层峰值**——包含中间 buffer/workspace

## 4. 为什么 Parameters ≠ Runtime Memory？

实测发现**系统性低估**（校准前 weight 平均误差 27%）：
- **FLOAT16 GEMM 权重**：flatten≥32768 时 classifier 权重存 FP16（2 bytes），
  int8 估算只算 1 byte → **低估 ~1.85 倍**（6 个模型实测 w_ratio 1.69~1.87）
- internal buffer / workspace / padding：实测高于单层 activation 估算

**分组证据**（10 模型实测）：

| 路径 | flatten | weight 实测/估算 | 结论 |
|------|---------|----------------|------|
| int8 | ≤16384 | 1.00~1.01 | 理论估算**几乎完美** |
| FLOAT16 GEMM | ≥32768 | 1.69~1.87 | 系统性低估 |

## 5. 哪些 Architecture Factors 对 Memory 最重要？

| 因素 | 影响 | 实测证据 |
|------|------|---------|
| **Flatten 维度** | 决定是否触发 FLOAT16 GEMM（weight ×~1.85） | 6/10 模型 |
| 输入分辨率 | 决定 activation 大小（当前 64×64 下影响小） | peak 131~262KB |
| 首层通道 | 决定峰值 activation | 32ch vs 64ch 差 2 倍 |
| Params | weight 主导项（当前规模占 >85%） | corr=0.999 |

## 6. Estimated 与实际 RK3566 Memory 是否一致？

**校准前**：Total 平均误差 33.5%（系统性低估）。
**校准后**（分段线性，按 flatten 阈值分组）：

```
int8 路径:     M_runtime ≈ 0.951 × M_est + 170,516   R²=0.986  MAPE=2.6%
fp16-gemm 路径: M_runtime ≈ 1.981 × M_est − 304,324   R²=0.995  MAPE=2.3%
```

**校准后平均误差 2.4%（全部 ±5% 内）**，模型见 `models/rknn_memory_v1.json`。

## 7. 误差来自哪里？

1. **FLOAT16 GEMM 权重**（flatten≥32768）：int8 估算的 1 byte vs 实际 2 bytes → 主误差源
2. **internal buffer 超出单层 activation**：RKNN 运行时中间张量 > 最大特征图
3. **weight padding/alignment**：layout 转换产生的少量额外空间

## 8. 当前 Memory Estimator 是否足以用于 Hardware-aware Search？

**是**。校准后 MAPE=2.4%，可作为 Search 的内存约束（`evaluate_candidate` 已输出
`predicted_runtime_memory_bytes`）。注意：校准基于 10 个模型（初步校准，
如需更高可信度可增加样本）。

## 9. 与 Latency Predictor 相比，Memory Model 可靠程度？

| | Latency (v1) | Memory (v1) |
|--|-------------|-------------|
| 验证指标 | R²=0.972, MAPE=9.2% | 校准后 MAPE=2.4% |
| 样本 | 22 训练 + 12 验证 | 10 校准 |
| 特征 | conv_macs + linear_macs | flatten 阈值分组 + 估算内存 |
| 结论 | 可用（预留余量） | 可用（误差更小） |

Memory 模型校准后误差更小（2.4% vs 9.2%），但校准样本更少（10 vs 34），
两者都是第一版，用于 Search 预筛是可靠的。

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `results/memory_profiles.csv` | 10 代表模型估算 profile（int8/fp16） |
| `results/memory_benchmark.csv` | 10 模型估算 vs eval_memory 实测 |
| `models/rknn_memory_v1.json` | 分段校准模型参数 |
| `docs/figures/params_vs_memory.png` | Params vs Weight/Estimated |
| `docs/figures/activation_vs_memory.png` | Weight vs Activation 分解 |
| `docs/figures/estimated_vs_measured_memory.png` | 估算 vs 实测 |
| `docs/figures/memory_prediction_error.png` | 校准前后误差 |

## 复现

```bash
python analyze_memory.py        # 静态分析 + profile (无需板子)
python benchmark_memory.py      # eval_memory 实测 (需板子)
python fit_memory_model.py      # 拟合校准模型 -> rknn_memory_v1.json
```
