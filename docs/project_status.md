# Project Status（项目进度）

跨平台（RK3566 + H618）统一进度视图。RK3566 详细进度见
`rk3566/docs/PROJECT_PROGRESS.md`（含各阶段产出明细）。

## Overall Progress

```
阶段 0-5: RK3566 (部署→搜索→分析→建模→内存)  ✅ 完成
阶段 6-8: H618 (环境→benchmark→建模→切分可行性) ✅ 完成
阶段 9:   Accuracy 汇合 → Pareto Search          🔶 进行中
阶段 10:  通信 / 异构流水线                        ⬜ 计划
```

## Completed

- ✅ CNN Generator（参数化：depth/channels/kernel/pool）
- ✅ GTSRB 模型训练/评估（10 模型搜索，精度 94.4%~97.9%）
- ✅ RK3566 latency benchmark（22 模型 + 12 验证）
- ✅ RK3566 memory 分析（eval_memory 实测校准，10 模型）
- ✅ RK3566 latency/memory predictor（工程化 hpm 包）
- ✅ H618 NCNN 环境搭建（NDK→pnnx→NCNN→板端）
- ✅ H618 NCNN benchmark（25 模型 CPU/Vulkan）
- ✅ H618 CPU/Vulkan 对比（CPU 全面优于 Vulkan）
- ✅ H618 CPU latency predictor v2（5-fold CV MAPE 6.5%）
- ✅ Cross-hardware 分析（RK3566 vs H618）
- ✅ Flatten/Linear 受控实验（FD4K~FD64K）
- ✅ Partition 可行性预研（结论：单模型无延迟收益）

## In Progress

- 🔶 Accuracy / 训练数据汇合（队友提供 model_id → accuracy）
- 🔶 Hardware-aware Model Search（evaluate_candidate 已就绪，等 accuracy）

## Planned

- ⬜ Pareto 模型选择（Accuracy × Latency × Memory）
- ⬜ 通信 benchmark（若做异构：USB/网络 transfer 实测）
- ⬜ 异构流水线 / 资源分配（多模型并发场景）
- ⬜ 最终系统验证

## 当前数据资产

| 数据 | 规模 | 位置 |
|------|------|------|
| GTSRB 训练 | 10 模型精度 | rk3566/results/results.csv |
| RK3566 latency | 22 模型实测 | rk3566/results/benchmark_results.csv |
| RK3566 memory | 10 模型实测 | rk3566/results/memory_benchmark.csv |
| H618 latency | 25 模型实测 | h618/results/h618_dataset.csv |
| 验证集 | 12 独立模型 | rk3566/results/validation_benchmark.csv |
