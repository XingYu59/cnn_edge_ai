# 参数化 CNN 生成与 GTSRB 自动训练

赛道 A · 模块一（受限条件下 AI 模型训练与最优模型搜索）的基础实验程序：
**参数 → CNN → GTSRB → 训练 → 评估 → 结果记录** 全链路。

> 本仓库代码 **Windows / Linux 均可运行**。下文以 Windows 为例给出完整的新手环境搭建步骤，
> Linux 用户可参考文末的"Linux 快速参考"。

---

## 一、环境搭建（Windows，一步步来）

### 第 1 步：安装 Python 3.10

1. 打开浏览器访问: https://www.python.org/downloads/release/python-31011/
2. 往下滚，找到 **Windows installer (64-bit)** 并下载
3. 双击安装，**务必勾选底部 "Add Python 3.10 to PATH"**（非常重要！）
4. 一路 Next 完成安装

验证：打开 **命令提示符**（Win 键 → 输入 `cmd` 回车），输入：
```cmd
python --version
```
应显示 `Python 3.10.x`。如果提示"不是内部或外部命令"，说明没勾选 PATH，重装一次并勾选。

### 第 2 步：下载本项目代码

```cmd
git clone https://github.com/XingYu59/cnn_rk3566.git
cd cnn_rk3566
```
（没装 git 的话，也可以直接到 GitHub 页面点绿色 `Code` → `Download ZIP`，解压后进入文件夹）

### 第 3 步：创建虚拟环境（隔离依赖，避免污染系统）

在项目目录（`cnn_rk3566` 文件夹）里执行：
```cmd
python -m venv venv
venv\Scripts\activate
```
激活成功后，命令行**最前面会出现 `(venv)`**，像这样：
```
(venv) C:\Users\你的名字\cnn_rk3566>
```

### 第 4 步：安装依赖

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas pyarrow pyyaml
```
> - 上面的 torch 是 **CPU 版**，任何电脑都能装能跑（训练会慢一些，但能用）。
> - 如果你的电脑有 NVIDIA 显卡（RTX 20/30/40 系列），可以把第一行换成
>   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` 用 GPU 加速。
> - 国内网络如果下载慢，可在命令末尾加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### 第 5 步：下载数据集（约 312MB）

在项目目录执行（Windows 10 及以上自带 curl）：
```cmd
mkdir data\GTSRB
curl -L -o data\GTSRB\train.parquet "https://huggingface.co/datasets/bazyl/GTSRB/resolve/main/data/train-00000-of-00001-dc762c064c221993.parquet"
curl -L -o data\GTSRB\test.parquet  "https://huggingface.co/datasets/bazyl/GTSRB/resolve/main/data/test-00000-of-00001-747a54d4a6461a97.parquet"
```
> 如果 curl 失败，就手动下载：浏览器打开 https://huggingface.co/datasets/bazyl/GTSRB
> → 进入 `data/` 文件夹 → 下载两个 `.parquet` 文件 → **改名为** `train.parquet` 和 `test.parquet`
> → 放进 `cnn_rk3566\data\GTSRB\` 文件夹。

### 第 6 步：验证安装成功

```cmd
python main.py --mode analyze --config configs/model.yaml
```
应打印出模型结构、参数量、MACs、Feature Map 表格。看到这些就说明环境 OK 了！

---

## 二、快速开始

```cmd
rem 1) 静态分析（无需数据集, 看模型结构与参数量）
python main.py --mode analyze --config configs/model.yaml

rem 2) 训练单个模型（约 30 分钟~1 小时, CPU 版）
python main.py --mode train --config configs/model.yaml

rem 3) 搜索多个候选模型（每跑一个都要时间, 见下节）
python main.py --mode search --config configs/search_space.yaml
```

---

## 三、你的任务：扩大模型搜索

**目标**：在 `configs/search_space.yaml` 里增加更多候选结构，训练后对比精度，找出最优模型。

### 第 1 步：看懂搜索空间

`configs/search_space.yaml` 内容：
```yaml
search_space:
  depth: [3, 4, 5, 6]        # 网络深度（卷积层数）
  kernel_size: [3, 5]        # 卷积核大小
  channels:                  # 每层通道数（长度必须等于某个 depth，否则被跳过）
    - [16, 32, 32]
    - [16, 32, 64]
    - [32, 32, 64, 64]
    - [32, 64, 64, 128]
    - [32, 32, 64, 64, 128]
```

### 第 2 步：如何扩展（三个方向任选）

| 方向 | 怎么改 | 例子 |
|------|--------|------|
| 更深 | 增加 `depth` 值和对应长度的 `channels` | 加 `depth: 6`，加 `[32, 32, 64, 64, 128, 128]` |
| 更宽 | 增加 `channels` 组合（16/32/64/128 内取值，逐级增宽） | 加 `[64, 64, 128, 128]`（配 depth 4） |
| 更大核 | 增加 `kernel_size` 值（注意代码只支持 3/5） | 目前只支持 3 和 5 |

**规则**：`channels` 的长度必须与某个 `depth` 相等，否则该组合会被自动跳过（程序会提示）。

### 第 3 步：训练并查看结果

```cmd
python main.py --mode search --config configs/search_space.yaml
```
- 每个模型训练完后自动写入 `results/results.csv`
- 建议一次只加 5~10 个新模型，训练时间可控

### 第 4 步：用 Excel 对比结果

打开 `results/results.csv`（右键 → 用 Excel/记事本打开），对比：
- `test_accuracy`：测试精度（越高越好）
- `parameters` / `macs`：模型大小和计算量（越小越适合部署）
- **理想模型 = 精度高 + 参数量小 + MACs 小**

---

## 四、项目结构

```
cnn_rk3566/
├── main.py                  # 入口: train / analyze / search 三种模式
├── modules/
│   ├── generator.py         # CNN 生成器: 配置 -> 模型
│   ├── dataset.py           # GTSRB 数据读取
│   ├── analyzer.py          # 参数量 / MACs / Feature Map / 逐层分析
│   ├── trainer.py           # 训练 / 验证 / 测试
│   └── search.py            # 候选生成 + 硬件感知过滤
├── configs/
│   ├── model.yaml           # 单模型训练配置
│   └── search_space.yaml    # ★ 搜索空间（你的任务就是改这个）
├── convert_to_rknn.py       # 模型转 RKNN（部署到 RK3566）
├── verify_gtsrb.py          # 板端精度验证 + 测速
├── benchmark_rknn3566.py    # 统一 RK3566 benchmark（warmup+多次+逐层）
├── batch_convert_experiments.py  # 批量转换实验模型（随机权重）
├── controlled_experiments.py    # 4 组控制变量实验定义（22 模型）
├── analyze_controlled.py        # 相关性 + 回归 + 图 + 实验报告
├── validation_models.py         # 12 个独立验证模型
├── validate_latency_model.py    # Latency 模型验证全流程
├── generate_validation_report.py # 重新生成验证报告
├── demo_hpm.py                  # 性能模型验证与演示
├── analyze_memory.py            # 内存静态分析（代表模型 profile）
├── benchmark_memory.py          # eval_memory 实测（需板子）
├── fit_memory_model.py          # 内存校准模型拟合 → rknn_memory_v1.json
├── hpm/                         # ★ 硬件性能模型包（可调用）
│   ├── architecture.py          # 静态架构分析
│   ├── latency.py               # latency 预测（加载/拟合/预测）
│   ├── memory.py                # 内存估算 + 校准预测
│   ├── filter.py                # 硬件约束过滤
│   └── pipeline.py              # evaluate_candidate 搜索接口
├── models/rknn_latency_v1.json  # ★ latency 模型参数（实验拟合）
├── models/rknn_memory_v1.json   # ★ 内存校准模型参数（eval_memory 实测校准）
├── data/GTSRB/              # 数据集（自己下载, 不进 git）
├── results/                 # 训练/benchmark/验证/算子数据 CSV
└── docs/                    # ★ 技术文档与实验报告（见下方索引）
```

## 文档索引（重要，先读这个）

| 文档 | 内容 |
|------|------|
| [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) | 五阶段进度总览 + 核心成果 |
| [docs/rknn_performance_model.md](docs/rknn_performance_model.md) | **性能模型工程化报告**（latency/memory/filter/pipeline） |
| [docs/rknn_memory_analysis.md](docs/rknn_memory_analysis.md) | **内存分析报告**（估算 vs eval_memory 实测校准） |
| [docs/rknn_latency_model_validation_report.md](docs/rknn_latency_model_validation_report.md) | **RK3566 延迟模型验证报告（核心）** |
| [docs/controlled_benchmark_report.md](docs/controlled_benchmark_report.md) | 22 模型控制变量实验报告 |
| [docs/findings/01_float16_gemm_classifier_bottleneck.md](docs/findings/01_float16_gemm_classifier_bottleneck.md) | FINDING-001: classifier 瓶颈 |
| [docs/analysis/01_macs_vs_latency_analysis.md](docs/analysis/01_macs_vs_latency_analysis.md) | ANALYSIS-001: MACs vs 时延 |

> **核心结论**：RK3566 延迟可用 `T(us) ≈ 3.02×ConvMACs(M) + 980.5×LinearMACs(M) + 32` 预测（验证 R²=0.972, MAPE=9.2%）；**flatten 维度须 ≤16384**（≥32768 会触发 FLOAT16 GEMM 瓶颈）。
> 使用方式：`from hpm.pipeline import evaluate_candidate; evaluate_candidate(config, constraints)`

---

## 五、技术说明（队友了解即可）

### CNN 结构

每个 block: `Conv2d → BatchNorm2d → ReLU → (MaxPool2d, 每隔一个 block)`，
最后 Flatten → Linear。classifier 输入维度自动推断，不硬编码。

### GTSRB 数据

- 43 类交通标志，训练 39209 张 / 测试 12630 张
- 统一 Resize 到 64×64，归一化用 GTSRB 统计值
- 训练/验证按类别**分层**划分（每类 20% 做验证）

### 数据增强（为什么不用翻转）

交通标志的**语义对翻转敏感**："向左"标志翻转后变成"向右"。
所以只用旋转（±15°）和颜色抖动，**不用水平/垂直翻转**。

### 指标含义

- **Parameters**：模型权重总个数
- **MACs**：一次推理的理论乘加运算量（**不是真实延迟**，真实延迟需上硬件实测）
- **Model Size**：FP32 存储大小 = 参数 × 4 bytes

---

## 六、常见问题（FAQ）

**Q: `python` 不是内部或外部命令？**
A: 安装 Python 时没勾选 "Add to PATH"，重装并勾选，或手动把 Python 加入环境变量。

**Q: 激活 venv 提示"禁止运行脚本"？**
A: 用管理员身份打开 PowerShell，执行 `Set-ExecutionPolicy RemoteSigned` 允许脚本运行，然后重试。

**Q: 训练很慢？**
A: CPU 版就是慢（每个模型 30 分钟~1 小时）。有 NVIDIA 显卡就装 cu121 版 torch（见第 4 步注释）。

**Q: 下载数据集失败/超时？**
A: 换手动浏览器下载方式（见第 5 步），或挂代理后重试。

**Q: 搜索结果里 depth=6 没有模型？**
A: 正常——`channels` 里没有长度为 6 的组合，被程序自动跳过了。想用 depth=6 就加一个 6 元素的 channels。

---

## 七、Linux 快速参考（给负责 RKNN 部署的同学）

```bash
# 环境（本机已有）
python3 -m venv /home/xing/venvs/gtsrb     # 训练环境 (torch 2.11+cu128, GPU)
python3 -m venv /home/xing/venvs/rknn      # RKNN 转换环境 (torch 2.4 + rknn-toolkit2)

# 训练 / 分析 / 搜索
/home/xing/venvs/gtsrb/bin/python main.py --mode train   --config configs/model.yaml
/home/xing/venvs/gtsrb/bin/python main.py --mode analyze --config configs/model.yaml
/home/xing/venvs/gtsrb/bin/python main.py --mode search  --config configs/search_space.yaml

# 转 RKNN（部署到 RK3566）
/home/xing/venvs/rknn/bin/python convert_to_rknn.py \
    --ckpt results/<模型>_best.pt \
    --depth <d> --channels <c1,c2,...> --kernel-size <k> \
    --out models/<name>.rknn --verify
```

---

## 八、结果记录说明

`results/results.csv` 每行一个模型：
`model_id, depth, channels, kernel_size, input_size, parameters, macs,
 model_size_MB, best_validation_accuracy, test_accuracy, training_time`

目前仓库里的 `results.csv` 是 10 个基线模型的正式训练结果（30 epochs, GPU），
你的扩展结果会**追加**在后面，最终一起用于模块二（硬件性能建模）。

---

## 九、results 数据文件与图

| 文件 | 内容 |
|------|------|
| `results/results.csv` | 训练精度（10 模型） |
| `results/benchmark_results.csv` | **22 模型** RK3566 实测（静态+延迟+算子分解） |
| `results/validation_benchmark.csv` | 12 独立验证模型实测 |
| `results/latency_model_results.csv` | M1/M2 预测 vs 实测 |
| `results/memory_benchmark.csv` | 10 模型估算 vs eval_memory 实测 |
| `results/memory_profiles.csv` | 内存估算 profile |
| `results/operator_database.csv` | conv/gemm 算子聚合 |
| `docs/figures/` | 分析图（索引见 `../docs/figures.md`） |
| `models/rknn_latency_v1.json` / `rknn_memory_v1.json` | 性能模型参数 |

**核心结论速记**：`T(us)=3.02·Conv+980.5·Linear+32`（val R²=0.972, MAPE 9.2%）；
flatten≥32768 触发 FLOAT16 GEMM 惩罚；内存校准后 MAPE 2.3-2.6%。

**子项目上级**：仓库总览见 [`../README.md`](../README.md)，
跨平台进度/结论见 [`../docs/`](../docs/)。

> 本文件主要面向 Windows 队友（环境搭建/扩搜索）；完整技术细节在各 docs 报告。
