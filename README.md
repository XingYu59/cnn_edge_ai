# 异构边缘 AI 部署（RK3566 NPU + H618）

基于 **GTSRB 交通标志识别** 的异构边缘 AI 课程设计（赛道 A）统一仓库。

## 1. 项目简介

用参数化 CNN Generator 生成不同结构的 CNN，在 GTSRB（43 类）上训练，
并部署到两块国产异构平台实测性能，研究**模型结构 → 硬件执行**的关系，
最终支撑 Hardware-aware Model Search 与异构部署决策。

## 2. 项目目标

1. 模型自动生成、训练与搜索（模块一）
2. RK3566 NPU / H618 双平台性能建模（latency + memory）
3. 验证"MACs ≠ latency"，建立可调用的性能预测器
4. 评估异构切分可行性（Flatten/Linear 结构敏感性）
5. 最终：Accuracy × Latency × Memory 多目标选优

## 3. 当前技术路线

```
PyTorch CNN Generator
   ├── GTSRB 训练 → 精度
   ├── RK3566: RKNN (NPU, INT8)
   └── H618: ONNX → pnnx → NCNN (CPU/Vulkan)
              ↓
        静态特征 (MACs/Conv/Linear/Flatten/Params)
              ↓
    Latency/Memory Predictor (实测校准)
              ↓
       Hardware-aware 决策
```

## 4. 硬件平台

| 平台 | 芯片 | Runtime | 备注 |
|------|------|---------|------|
| rk3566/ | RK3566 NPU | RKNN-Toolkit2 2.3.2 / librknnrt 1.5.2 | NPU 900MHz, INT8 |
| h618/ | Cortex-A53×4 + Mali-G31 | NCNN 20260526 | Android 31, armeabi-v7a |

## 5. 模型与数据集

- 数据集：GTSRB（43 类交通标志，39209 训练 / 12630 测试，64×64 输入）
- 模型：参数化 CNN（Conv→BN→ReLU→Pool），depth 3~7, channels 16~128, kernel 3/5
- 实验集：**25 个模型**（14 legacy + 11 受控扩展），双平台同结构

## 6. 当前完成情况

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | RKNN 部署链路 (resnet18) | ✅ |
| 1 | CNN Generator + GTSRB 训练搜索 (10 模型) | ✅ |
| 2 | RK3566 控制变量实验 (22 模型) + FINDING-001 | ✅ |
| 3 | RK3566 latency 模型验证 (12 独立模型) | ✅ |
| 4 | RK3566 latency/memory predictor 工程化 (hpm 包) | ✅ |
| 5 | RK3566 memory 实测校准 (eval_memory) | ✅ |
| 6 | H618 NCNN 环境 + 最小链 | ✅ |
| 7 | H618 benchmark (25 模型 CPU/Vulkan) + predictor | ✅ |
| 8 | 异构切分可行性 (Flatten/Linear 效应) | ✅ |
| 9 | Accuracy 汇合 + Hardware-aware Pareto Search | 🔶 进行中 |
| 10 | 真实通信 / 异构流水线 | ⬜ 计划 |

详细进度见 [docs/project_status.md](docs/project_status.md)

## 7. 核心实验结果（摘要）

详细见 [docs/experimental_results.md](docs/experimental_results.md)

### RK3566 (NPU)
- 22 模型实测；**总 MACs 不能单独解释 latency**（M1 R²=0.55），需区分算子类型
- Latency predictor（实测校准）：`T(us)=3.02·Conv(M)+980.5·Linear(M)+32`
  → 验证集 R²=0.972, MAPE=9.2%
- Memory：INT8 路径估算误差 <1%；FLOAT16 GEMM 路径校准后 MAPE=2.3-2.6%
- **FINDING-001**：flatten ≥32768 触发 FLOAT16 GEMM 惩罚（Linear 开销为 Conv 的 ~325×）

### H618 (NCNN)
- 25 模型 CPU/Vulkan 实测；**CPU 全面优于 Vulkan**（Mali-G31 低端 GPU + 调度开销）
- CPU predictor v2（5-fold CV）：Conv+Linear 模型 MAPE=6.5%
- 双平台：H618 CPU 平均慢 RK3566 NPU ~12×（Vulkan ~28×）

## 8. 当前研究结论

1. **MACs ≠ latency**（RK3566 上明显；H618 CPU 上 MACs 相关性高达 0.99）
2. **算子类型是 RK3566 延迟的主要调节因素**（FLOAT16 GEMM 路径）
3. **RK3566 绝对延迟全面低于 H618**（conv 30×, linear 1.2×）
4. Linear-heavy 模型**相对**降低 RK3566 优势，但 **H618 无绝对优势**（relative ≠ absolute）
5. 单模型异构切分**无延迟收益**（T_hetero > T_RK 必然）；价值仅在并发/资源分配

## 9. 仓库结构

```
cnn/
├── rk3566/              # RK3566 全流程 (训练/转换/benchmark/建模)
│   ├── main.py, modules/   # CNN Generator + 训练
│   ├── hpm/                # 硬件性能模型包 (latency/memory/filter)
│   ├── convert_to_rknn.py, verify_gtsrb.py
│   ├── benchmark_rknn3566.py, analyze_*.py
│   ├── results/, models/, docs/
└── h618/                # H618 NCNN 全流程
    ├── ncnn_test/           # 最小运行链
    ├── ncnn_bench/          # benchmark 工程
    ├── convert_to_ncnn.py, analyze_*.py, fit_*.py
    ├── models/, results/, docs/
├── docs/                # ★ 跨平台统一文档 (本目录)
├── .gitignore
```

## 10. 如何复现

各子项目 README/文档含完整命令。关键入口：
- RK3566 转换/部署：`rk3566/README.md`
- RK3566 建模：`rk3566/docs/rknn_performance_model.md`
- H618 建模：`h618/docs/h618_ncnn_performance.md`
- 双平台对比：`docs/cross_hardware_analysis.md`

## 11. 当前限制

- 实验基于 GTSRB 单数据集、64×64 单输入尺寸
- RK3566 部分扩展模型为 predictor 预测（标注 predicted，非实测）
- 通信未实测（transfer 为理论估算）
- Accuracy 数据待队友汇合（当前 pipeline 无精度维度）

## 12. 下一步计划

见 [docs/next_steps.md](docs/next_steps.md)——核心：Accuracy 汇合 → Hardware-aware Pareto Search；若做异构则需真实通信 benchmark。
