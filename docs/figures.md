# Figures 索引

实验图分散在各子项目 `docs/figures/`，此处汇总关键图。

## RK3566（rk3566/docs/figures/）

| 图 | 内容 | 用途 |
|----|------|------|
| fig_controlled_1to6.png | 控制变量实验 (MACs/Params/Flatten vs latency) | 性能相关性 |
| fig5_layer_latency_breakdown.png | int8 conv vs FLOAT16 GEMM 耗时分解 | FINDING-001 证据 |
| fig_m2_validation.png | M2 预测 vs 实测 (R²=0.972) | Latency 模型验证 |
| fig_prediction_error.png | M1/M2 误差对比 | 模型对比 |
| estimated_vs_measured_memory.png | 内存估算 vs eval_memory 实测 | Memory 校准 |
| params_vs_memory.png | Params vs Weight/Estimated | Memory 分析 |
| rk3566_vs_h618_latency.png | RK3566 vs H618 (h618 目录) | 跨硬件对比 |

## H618（h618/docs/figures/）

| 图 | 内容 | 用途 |
|----|------|------|
| h618_cpu_vulkan_analysis.png | CPU vs Vulkan + ratio | Backend 选择 |
| h618_flatten_dual_platform.png | 双平台 Flatten vs latency | Flatten 效应 |
| h618_cpu_predicted_vs_measured.png | CPU predictor 验证 | Predictor v2 |
| h618_vulkan_predicted_vs_measured.png | Vulkan predictor 验证 | Vulkan 建模 |
