# 项目交接总结（HANDOFF）

> 供新会话/新协作者无缝续接。最后更新：2026-09-02

## 1. 项目是什么

**基于 GTSRB（43 类交通标志）的异构边缘 AI 课程设计（赛道 A）**：
参数化 CNN Generator → 训练搜索 → 部署到两块国产平台（RK3566 NPU + H618）
实测性能 → 建立硬件感知性能模型 → 为 Hardware-aware Model Search 与异构部署决策提供依据。

GitHub: `XingYu59/cnn_edge_ai`（monorepo，含 rk3566/ + h618/ 两个子项目）
本地: `~/RK3566_dev/cnn/`

## 2. 环境（关键路径，勿重建）

| 项 | 路径/值 |
|----|---------|
| RKNN venv | `/home/xing/venvs/rknn`（torch 2.4.0 + rknn-toolkit2 2.3.2 + onnx 1.16.1 + pandas） |
| 训练/分析 venv | `/home/xing/venvs/gtsrb`（torch 2.11.0+cu128，GPU RTX 5070 Ti 用） |
| K11C (RK3566) | librknnrt 1.5.2, NPU 900MHz, adb 设备 `rk3566_t` |
| H618 | Android 31, armeabi-v7a, 4×Cortex-A53, Mali-G31, adb 设备 `QUAD_CORE_H618_p2` |
| NCNN 预编译 | `~/RK3566_dev/ncnn-20260526-android-vulkan`（armeabi-v7a） |
| NCNN 源码 | `~/RK3566_dev/ncnn-20260526-src`（onnx2ncnn 用，但实际用了 pnnx） |
| NDK | `~/Android/android-ndk-r29` |
| pnnx | `/home/xing/venvs/rknn/bin/pnnx`（pip 安装，20260526） |
| 数据集 | GTSRB parquet 在 `cnn/rk3566/data/GTSRB/`（HuggingFace bazyl/GTSRB，官方 zip 失效） |

**注意**：adb 同一时刻只能连一块板（RK3566 或 H618），切板需换 USB。沙箱环境无 GPU 设备，训练/实测需在用户终端执行或带权限。

## 3. 完成的工作（9 个阶段 ✅）

| # | 阶段 | 要点 |
|---|------|------|
| 0 | RKNN 部署链路 (resnet18) | 转换→上板→8.24ms 实测 |
| 1 | CNN Generator + GTSRB 训练 | 10 模型搜索，精度 94.4~97.9%，最优 d5_k3 (97.67%) |
| 2 | RK3566 控制变量实验 | 22 模型，FINDING-001 |
| 3 | RK3566 latency 模型验证 | 12 独立模型 |
| 4 | 工程化 hpm 包 | architecture/latency/memory/filter/pipeline |
| 5 | RK3566 memory 实测校准 | eval_memory，10 模型 |
| 6 | H618 NCNN 环境 | NDK→pnnx→NCNN→最小链 |
| 7 | H618 benchmark | 25 模型 CPU/Vulkan + predictor |
| 8 | 异构切分可行性 | Flatten/Linear 效应 + partition 分析 |

文档整理阶段已完成（README/docs 体系，commit 2ed818f/4dac1f1 已 push）。

## 4. 核心实验结果（数字，勿虚构）

### RK3566 (NPU)
- 22 模型实测 + 12 验证
- **Latency predictor**：`T(us)=3.02×Conv(M)+980.5×Linear(M)+32` → val R²=0.972, MAPE=9.2%
- **Memory**：校准后 int8 路径 MAPE 2.6% / fp16-gemm 路径 2.3%
- **FINDING-001**：flatten≥32768 → FLOAT16 GEMM 惩罚（Linear 开销 = Conv 的 ~325×）
- 总 MACs 不能单独解释延迟（M1 R²=0.55）

### H618 (NCNN)
- 25 模型实测（14 legacy + 11 受控 FD/XL/MD）
- **CPU 全面优于 Vulkan**（Mali-G31，Vulkan 慢 1.4~2.5×，无 crossover）
- **CPU predictor v2**（5-fold CV）：M2 Conv+Linear → R²=0.938, MAPE=6.5%
- CPU 上 MACs 相关 0.99（无 RK3566 的 FLOAT16 GEMM 惩罚）

### 双平台 / 切分
- H618 CPU 平均慢 RK3566 ~12×（Vulkan ~28×）
- RK3566 linear/conv 比值 325× vs H618 13×（H2 支持：H618 无特殊惩罚）
- **但 RK3566 绝对延迟全胜**（conv 30×、linear 1.2×）→ relative ≠ absolute
- **单模型异构切分无延迟收益**（T_hetero > T_RK 必然）→ 切分价值仅在并发/资源分配场景（未验证）

## 5. 代码结构速览

```
cnn/
├── README.md / docs/          # 总览 + 跨平台文档（进度/结果/对比/下一步/图索引）
├── rk3566/                    # RK3566 全流程
│   ├── main.py modules/       # Generator/训练/搜索
│   ├── hpm/                   # ★ 可调用性能模型包: evaluate_candidate(config, constraints)
│   ├── convert_to_rknn.py     # checkpoint→.rknn（mean/std ×255 换算）
│   ├── benchmark_rknn3566.py  # 统一 benchmark（eval_mem/eval_perf/perf_debug）
│   ├── results/ models/*.json docs/
└── h618/                      # H618 NCNN
    ├── ncnn_test/ ncnn_bench/ # C++ 工程（NDK 编译）
    ├── convert_to_ncnn.py     # →ONNX→pnnx→ncnn param/bin（blob: in0/out0）
    ├── run_h618_bench.sh      # 批量 CPU/Vulkan benchmark
    ├── fit_h618_cpu_v2.py     # predictor v2 (5-fold CV)
    ├── results/ models/ docs/
```

## 6. 下一步（待办）

1. **等队友 accuracy 数据**（model_id → accuracy）
2. **Hardware-aware Pareto Search**：`evaluate_candidate`（latency/memory/params 已就绪）
   + accuracy → Pareto Front → 最优模型（RK3566 部署）
3. （可选/若做异构）真实通信 benchmark（FM 16KB~64KB 级别）→ 判断并发切分价值
4. 课程报告整理

## 7. 关键避坑记录（新会话必读）

- **沙箱无 GPU**：训练/实测脚本需在用户终端跑或带权限（danger-full-access）
- **RKNN 连板**：`load_rknn` 不能模拟器跑，须 `init_runtime(target='rk3566')`
- **onnx 必须 1.16.1**（rknn venv），1.22 会报 `onnx.mapping` 缺失
- **mean/std ×255 换算**：rknn.config 的 mean/std 是 0-255 域 = torch 训练值 ×255
- **模型二进制不进 git**（.rknn/.bin/checkpoint 忽略，可重新生成）
- **H618 模型 blob 名**：pnnx 转换后是 `in0`/`out0`（不是 input/output）
- **NCNN Vulkan 需先 create_gpu_instance**，net 析构先于 destroy_gpu_instance（否则 segfault）
- **当前 adb 可能连 H618**；切 RK3566 需换 USB
- 代理 127.0.0.1:7897 常不可达 → pip/gh 用 `env -u http_proxy...` 直连或清华镜像

## 8. 常用命令速查

```bash
# 转换 GTSRB 模型（rk3566 venv）
cd ~/RK3566_dev/cnn/rk3566
python convert_to_rknn.py --ckpt results/xxx_best.pt --depth 4 --channels 16,32,64,64 --kernel-size 3 --out models/xxx.rknn

# 硬件评估（不跑实机）
python -c "from hpm.pipeline import evaluate_candidate; print(evaluate_candidate({'depth':4,'channels':[32,32,64,64],'kernel_size':3,'num_classes':43,'input_size':64}))"

# H618 转换 + benchmark
cd ~/RK3566_dev/cnn/h618
python convert_to_ncnn.py cnn_test    # → models/cnn_test.param/bin
./run_h618_bench.sh "cnn_test" 200    # 需 H618 连接

# 数据分析（gtsrb venv）
python make_h618_dataset.py && python fit_h618_cpu_v2.py
```
