# 项目进度总览

**项目**：基于 GTSRB 的 RK3566 异构边缘 AI 课程设计（赛道 A）
**更新日期**：2026-09-01
**仓库**：https://github.com/XingYu59/cnn_rk3566

---

## 一、三阶段进度

### ✅ 阶段 0：RKNN 基础部署链路（resnet18 示例）

| 项 | 结果 |
|----|------|
| 模型转换 | PyTorch → TorchScript → RKNN Toolkit2 → `.rknn` (int8) |
| PC 验证 | 模拟器 TOP5 = "space shuttle" 99.97% |
| RK3566 实测 | resnet18: **8.24ms** / FPS 121 |
| 环境攻坚 | torchvision 配套版本、onnx 1.16.1 兼容、权重缓存重定向 |

**产出**：`convert_to_rknn.py` 原型、`verify_rknn.py`、`benchmark_resnet18.py`

### ✅ 阶段 1：赛道 A 模块一（参数化 CNN + GTSRB 搜索）

| 项 | 结果 |
|----|------|
| 程序 | Generator/Dataset/Analyzer/Trainer/Search 五模块 + CLI |
| 训练 | cnn_test **96.87%**；10 候选搜索 **94.38%~97.85%** |
| 最优模型 | d5_k3 (118M MACs, **97.67%**) — 精度/效率拐点 |
| 上板 | cnn_d5_k3.rknn: 精度 **97.6%** / NPU **1.88ms** / FPS 533 |
| 环境 | 独立训练 venv (torch 2.11+cu128, RTX 5070 Ti) |

**产出**：`main.py` + `modules/`、`convert_to_rknn.py`、`verify_gtsrb.py`

### ✅ 阶段 2：硬件感知分析与控制变量实验（22 模型）

| 项 | 结果 |
|----|------|
| 发现 | FINDING-001: FLOAT16 Gemm classifier 瓶颈（flatten≥32768 触发） |
| 实验 | 4 组控制变量: A depth(5) / B kernel(4) / C featuremap(5) / D classifier(4) |
| 回归 | M1(总MACs) R²=0.550 vs M2(分算子) R²=0.987 |
| 关键发现 | ① 浅层大 kernel 更贵 ② flatten 决定 classifier 开销 ③ Linear 存在 int8 阈值 |

**产出**：`controlled_experiments.py`、`benchmark_rknn3566.py`、`analyze_controlled.py`、ANALYSIS-001

### ✅ 阶段 3：Latency Model 验证（12 独立模型）

| 项 | 结果 |
|----|------|
| 拟合 | M1: T=7.39·MACs+374 / M2: T=3.02·Conv+980·Lin+32 |
| **验证** | M1 val R²=**0.104** (MAPE 33%) vs **M2 val R²=0.972 (MAPE 9.2%)** |
| 结论 | **M2 泛化良好 (<10%) → 保留为主模型**（情况 A） |
| 里程碑 | V11 证明 M2 精确捕获 classifier 开销 (-1.3% vs M1 -60.6%) |

**产出**：`validation_models.py`、`validate_latency_model.py`、`latency_model_results.csv`、`operator_database.csv`、验证报告

### ✅ 阶段 4：工程化封装（hpm 包）

| 项 | 结果 |
|----|------|
| 架构分析 | `hpm/architecture.py`：纯静态分析，不依赖实机 |
| Latency 模型 | `hpm/latency.py` + `models/rknn_latency_v1.json`（真实实验拟合） |
| 内存估算 | `hpm/memory.py`（estimated 标记，非 Runtime 实测） |
| 约束过滤 | `hpm/filter.py`：latency/memory/params 约束 |
| Pipeline | `hpm/pipeline.py`：`evaluate_candidate`（搜索接口） |
| 验证 | predictor 复算 R²=0.9721 / MAE=112.8us / MAPE=9.2% ✅ |

**产出**：`hpm/` 包、`models/rknn_latency_v1.json`、`demo_hpm.py`、`docs/rknn_performance_model.md`

### ✅ 阶段 5：Memory 分析与校准（10 模型实测）

| 项 | 结果 |
|----|------|
| 内存估算 | `hpm/memory.py`：weight/每层 activation/peak/estimated（INT8/FP16） |
| 实测 | `eval_memory()`（`init_runtime(eval_mem=True)`）实测 10 代表模型 |
| **关键发现** | flatten≤16384 (int8) weight 误差 <1%；flatten≥32768 (FLOAT16 GEMM) 低估 1.85 倍 |
| **校准** | 分段线性校准：MAPE **33.5% → 2.4%**（int8 组 2.6% / fp16 组 2.3%） |
| 双预测器 | `evaluate_candidate` 输出 latency + runtime memory + feasible |

**产出**：`analyze_memory.py`、`benchmark_memory.py`、`fit_memory_model.py`、`models/rknn_memory_v1.json`、`docs/rknn_memory_analysis.md`

---

## 二、当前核心成果（可直接用于课程交付）

### 经验证的 RK3566 双预测器（Latency + Memory，已工程化）

```python
from hpm.pipeline import evaluate_candidate
r = evaluate_candidate(config, constraints)
# → predicted_latency_us / predicted_runtime_memory_bytes / feasible
```

```
Latency: T(us) ≈ 3.02×ConvMACs(M) + 980.5×LinearMACs(M) + 32.0
         R²=0.972, MAPE=9.2% (12 独立模型验证)
Memory:  分段校准 (flatten 阈值 16384)
         int8 路径 MAPE=2.6% / fp16-gemm 路径 MAPE=2.3%
         模型参数: models/rknn_latency_v1.json + rknn_memory_v1.json
```

### 关键结构约束（来自实测）

```
1. flatten 维度必须 ≤ 16384 (≥32768 触发 FLOAT16 GEMM, +700~1400us)
2. 浅层避免大 kernel (5×5 在浅层效率差)
3. 精度-效率最优: d5_k3 (118M MACs, 97.67%)
```

### 数据集资产

| 文件 | 内容 |
|------|------|
| `results/results.csv` | 10 个模型训练精度/参数/MACs |
| `results/benchmark_results.csv` | **22 个模型** RK3566 实测 (静态+延迟+算子分解) |
| `results/validation_benchmark.csv` | 12 个独立验证模型实测 |
| `results/latency_model_results.csv` | M1/M2 预测 vs 实测 |
| `results/operator_database.csv` | conv/gemm 算子聚合数据库 |
| `results/memory_profiles.csv` | 10 代表模型内存估算 (int8/fp16) |
| `results/memory_benchmark.csv` | 10 模型估算 vs eval_memory 实测 |

---

## 三、文档索引

| 文档 | 内容 |
|------|------|
| [README.md](../README.md) | Windows 队友环境搭建 + 使用指南 |
| [docs/findings/01_float16_gemm_classifier_bottleneck.md](findings/01_float16_gemm_classifier_bottleneck.md) | FINDING-001: classifier 瓶颈 |
| [docs/analysis/01_macs_vs_latency_analysis.md](analysis/01_macs_vs_latency_analysis.md) | ANALYSIS-001: 4 模型 MACs vs latency |
| [docs/controlled_benchmark_report.md](controlled_benchmark_report.md) | 22 模型控制变量实验报告 |
| [docs/rknn_latency_model_validation_report.md](rknn_latency_model_validation_report.md) | **M2 验证报告（核心）** |
| [docs/rknn_performance_model.md](rknn_performance_model.md) | **性能模型工程化报告**（latency/memory/filter/pipeline） |
| [docs/rknn_memory_analysis.md](rknn_memory_analysis.md) | **内存分析报告**（估算 vs eval_memory 实测校准） |

---

## 四、下一步

1. **等队友 accuracy 数据**（model_id → accuracy），汇合进 Pipeline
2. **Hardware-aware Model Search**：`generator → evaluate_candidate → 训练 → accuracy → Pareto Front`
   （evaluate_candidate 已具备 latency + memory + params 三重约束）
3. **队友扩搜索**：Windows 环境搭建见 README；避开 flatten≥32768 结构
4. **模块二建模**：延迟 + 内存模型均已校准（MAPE 9.2% / 2.4%），可进入正式建模
5. **模块三**（后续）：模型分割 + 星闪，基于逐层实测数据
