# 技术文档索引

项目技术发现、分析与实验报告的归档索引。
进度总览见 [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)。

## 进度总览

| 文档 | 内容 |
|------|------|
| [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) | 三阶段进度总览 + 核心成果 + 下一步 |

## 发现记录 (Findings)

| 编号 | 标题 | 日期 | 状态 |
|------|------|------|------|
| [FINDING-001](findings/01_float16_gemm_classifier_bottleneck.md) | FLOAT16 Gemm Classifier 瓶颈：浅层模型特征图过大导致 NPU 实测延迟异常 | 2026-09-01 | ✅ 已确认 |

## 分析报告 (Analysis)

| 编号 | 标题 | 日期 | 状态 |
|------|------|------|------|
| [ANALYSIS-001](analysis/01_macs_vs_latency_analysis.md) | MACs 与实际时延分析：回归 + 假设验证 + Q1-Q6（4 模型） | 2026-09-01 | ✅ |
| [Controlled Benchmark](controlled_benchmark_report.md) | 22 模型控制变量实验（A depth / B kernel / C featuremap / D classifier） | 2026-09-01 | ✅ |
| [Latency Model Validation](rknn_latency_model_validation_report.md) | **M2 验证报告**：12 独立模型，val R²=0.972 / MAPE=9.2% | 2026-09-01 | ✅ 情况A |

## 图表 (docs/figures/)

| 图 | 内容 |
|----|------|
| fig1234_metrics_vs_latency.png | MACs/Params/Classifier MACs/Flatten vs latency |
| fig5_layer_latency_breakdown.png | int8 conv vs FLOAT16 gemm 耗时分解 |
| fig_controlled_1to6.png | 控制变量实验图 1-6 |
| fig7_kernel_vs_latency.png | kernel 配置 vs latency |
| fig_validation_prediction.png | Fig V1: M1/M2 实测 vs 预测 |
| fig_prediction_error.png | Fig V2: 预测误差按模型 |
| fig_m2_validation.png | Fig V3: M2 验证 (R²=0.972) |

## 数据 (results/)

| 文件 | 内容 |
|------|------|
| results.csv | 10 模型训练精度 |
| benchmark_results.csv | 22 模型 RK3566 实测 |
| validation_models.csv / validation_benchmark.csv | 12 独立验证模型 |
| latency_model_results.csv | M1/M2 预测 vs 实测 |
| operator_database.csv | conv/gemm 算子聚合数据 |
| board_results.csv / layer_latency.csv | 早期板端数据 |

## 工程化封装 (hpm 包)

| 文档 | 内容 |
|------|------|
| [rknn_performance_model.md](rknn_performance_model.md) | **硬件性能模型工程化报告**（latency/memory/filter/pipeline） |
| `hpm/` 包 + `models/rknn_latency_v1.json` | 可调用的 Performance Model 基础设施 |

## 内存分析 (Memory)

| 文档 | 内容 |
|------|------|
| [rknn_memory_analysis.md](rknn_memory_analysis.md) | **内存分析报告**：估算 vs eval_memory 实测，分段校准 MAPE 33.5%→2.4% |
| `models/rknn_memory_v1.json` | 分段校准模型参数（flatten 阈值 16384） |
| `results/memory_benchmark.csv` | 10 模型估算 vs 实测 |
