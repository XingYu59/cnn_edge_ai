# H618 NCNN 性能测试子项目

RK3566 + H618 异构项目中的 **H618 侧**：在 H618（Allwinner，Cortex-A53×4 + Mali-G31）
上用 **NCNN** 跑 GTSRB CNN 模型，做 CPU/Vulkan 延迟 benchmark 与性能建模。

## 环境

```
H618: Android 31, armeabi-v7a, 4×Cortex-A53, Mali-G31, 4GB RAM
NCNN: ~/RK3566_dev/ncnn-20260526-android-vulkan (预编译, armeabi-v7a)
NDK : ~/Android/android-ndk-r29
转换: pnnx (pip install pnnx, 与 NCNN 同版本 20260526)
```

## 目录结构

```
cnn/h618/
├── ncnn_test/                  # ★ 最小运行链 (环境验证)
│   ├── CMakeLists.txt / main.cpp    # Hello NCNN + Vulkan 检查
│   ├── infer_mini.cpp               # 最小模型 CPU/Vulkan 推理
│   └── gen_mini_model.py            # 生成 mini.param/bin
├── ncnn_bench/                 # ★ benchmark 工程
│   ├── CMakeLists.txt / main.cpp    # h618_ncnn_bench (CPU/Vulkan)
│   └── build/                      # 编译产物 (不进 git)
├── convert_to_ncnn.py          # PyTorch→ONNX→pnnx→ncnn 模型转换
├── extra_experiments.py        # 受控扩展实验模型定义 (FD/XL/MD 组)
├── run_h618_bench.sh           # 批量 benchmark 收集脚本
├── make_h618_dataset.py        # 合并静态特征 + 实测 → h618_dataset.csv
├── analyze_h618.py             # CPU/Vulkan 相关性分析 + 图
├── analyze_flatten_effect.py   # Flatten/Linear 双平台效应 (H1/H2)
├── analyze_partition_candidates.py  # 异构切分可行性
├── fit_h618_latency_model.py   # CPU/Vulkan predictor v1
├── fit_h618_cpu_v2.py          # CPU predictor v2 (5-fold CV)
├── validate_h618_predictor.py  # predictor 验证图
├── compare_hardware.py         # RK3566 vs H618 对比
├── make_baseline.py            # baseline summary
├── models/                     # 25 模型 param/bin + predictor json
├── results/                    # benchmark/dataset CSV
└── docs/                       # 报告 + figures
```

## 各文件用途速查

| 文件 | 作用 | 何时用 |
|------|------|--------|
| `convert_to_ncnn.py` | 模型转 NCNN（复用 rk3566 Generator，随机权重测结构性能） | 加模型时 |
| `run_h618_bench.sh` | 批量跑 CPU+Vulkan benchmark → h618_latency.csv | 收集数据 |
| `make_h618_dataset.py` | 合并静态特征（MACs/Params/Flatten）到数据集 | 分析前 |
| `fit_h618_cpu_v2.py` | CPU predictor v2（25 模型 5-fold CV）→ json | 建模 |
| `analyze_flatten_effect.py` | 验证 Flatten 双平台敏感性（H1/H2） | 切分研究 |
| `compare_hardware.py` | RK3566/H618 延迟倍数 | 跨平台对比 |

## 工作流（复现）

```bash
# 1. 模型转换 (PC, rknn venv): 从 rk3566 Generator 生成结构 → ncnn param/bin
python convert_to_ncnn.py cnn_test d3_k3 ...   # models/*.param + *.bin

# 2. 编译 benchmark (需 NDK)
cd ncnn_bench
export ANDROID_NDK=$HOME/Android/android-ndk-r29
cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=android-31 \
  -Dncnn_DIR=$HOME/RK3566_dev/ncnn-20260526-android-vulkan/armeabi-v7a/lib/cmake/ncnn
cmake --build build

# 3. 部署 + benchmark (H618 已连 adb)
adb push build/h618_ncnn_bench /data/local/tmp/h618bench/
adb shell mkdir -p /data/local/tmp/h618bench/models
adb push models/*.param models/*.bin /data/local/tmp/h618bench/models/
./run_h618_bench.sh "cnn_test d3_k3 ..." 200   # → results/h618_latency.csv

# 4. 分析 (gtsrb venv)
python make_h618_dataset.py && python analyze_h618.py
python fit_h618_cpu_v2.py
```

> blob 命名：pnnx 转换后输入 `in0`、输出 `out0`（非 input/output）。
> 模型二进制（.bin）不进 git，可重新转换。

## 结果数据（results/）

| 文件 | 内容 |
|------|------|
| h618_latency.csv | 25 模型 × CPU/Vulkan 原始延迟（mean/median/min/max/std/p95） |
| h618_dataset.csv | 合并静态特征后的完整数据集 |
| h618_baseline_summary.csv | 14 模型双平台 baseline |
| partition_candidates.csv | 分段延迟理论估算 |

模型 predictor：`models/h618_cpu_latency_v2.json`（25 模型 5-fold CV, MAPE 6.5%）、
`h618_cpu_latency_v1.json`、`h618_vulkan_latency_v1.json`

## 关键结论（详见 docs/）

1. H618 CPU 全面优于 Vulkan（Mali-G31 低端 GPU），当前建模以 CPU 为主
2. H618 CPU 上 MACs 相关性 0.99（无 RK3566 的 FLOAT16 GEMM 惩罚）
3. H618 CPU 平均慢 RK3566 NPU ~12×；大 flatten 模型差距缩小（~6×）
4. 单模型异构切分无延迟收益（RK3566 绝对全胜）

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/h618_ncnn_performance.md](docs/h618_ncnn_performance.md) | H618 NCNN 性能建模报告 |
| [docs/h618_partition_feasibility.md](docs/h618_partition_feasibility.md) | 扩展实验 + 切分可行性（Q1-Q8） |
| [docs/figures/](docs/figures/) | 分析图 |
| 上级总览 | [../README.md](../README.md) / [../docs/project_status.md](../docs/project_status.md) |
