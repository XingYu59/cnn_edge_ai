# 技术发现记录：FLOAT16 Gemm Classifier 瓶颈

| 项 | 值 |
|----|----|
| 编号 | FINDING-001 |
| 日期 | 2026-09-01 |
| 状态 | ✅ 已确认（重测可复现） |
| 平台 | K11C / RK3566, librknnrt 1.5.2, NPU 900MHz |
| 相关代码 | `modules/generator.py`（池化策略）、`modules/analyzer.py`（MACs 统计） |

---

## 1. 摘要

在 RK3566 NPU 上板实测 4 个 GTSRB 模型时发现异常：**理论 MACs 最小的模型（d3_k3, 31.5M）实测 NPU 延迟反而最大之一（1.54ms），比 MACs 大 2.5 倍的 cnn_test（78M, 0.87ms）慢 76%**。

逐层耗时分析定位根因：**Generator 的池化策略导致浅层模型的最终特征图分辨率过大（32×32），flatten 后 classifier 输入高达 32768 维，RKNN 将该 Linear 转换为多个 FLOAT16 Gemm 算子，在 NPU 上执行效率极低（DDR 搬运主导）**。classifier 单独耗时约 900us，占全模型 88%，而 3 个卷积层仅占 12%。

**核心结论：NPU 实际延迟不能仅由总 MACs 预测，必须区分算子类型（int8 Conv 高效 vs FLOAT16 Gemm 低效）。**

---

## 2. 实测数据总览

| 模型 | 参数 | MACs（理论） | 训练精度 | 板端精度 | NPU 延迟 | 实际吞吐 |
|------|------|-------------|---------|---------|---------|---------|
| d3_k3 (31.5M) | 1.42M | 31.5M | 94.38% | 94.5% | **1.54 ms** | 0.020 TOPS |
| cnn_test (78M) | 765K | 78M | 96.87% | 97.1% | **0.87 ms** | 0.089 TOPS |
| d5_k3 (118M) | 1.55M | 118M | 97.67% | 97.6% | **1.88 ms** | 0.063 TOPS |
| d5_k5 (326M) | 1.80M | 326M | 97.85% | 98.2% | **2.46 ms** | 0.132 TOPS |

> 注：以上均含量化后精度（int8），全量/2000 张测试集实测。原始数据见 `results/board_results.csv`。

**异常点**：d3_k3 的 MACs 只有 cnn_test 的 40%，实测延迟却是其 1.77 倍。

---

## 3. 定位过程

### 3.1 方法

对可疑模型开启 `init_runtime(target='rk3566', perf_debug=True)` 后调用 `eval_perf()`，
获取逐算子耗时（DDR Cycles / NPU Cycles / Time(us) / RW(KB)）。

### 3.2 d3_k3 逐层耗时（关键行）

```
ID  OpType        DataType  InputShape                     OutputShape       Time(us)  RW(KB)
1   ConvRelu      UINT8     (1,3,64,64),(16,3,3,3)         (1,16,64,64)      90        78
2   ConvRelu      INT8      (1,16,64,64),(32,16,3,3)       (1,32,64,64)      116       198
4   ConvRelu      INT8      (1,32,32,32),(32,32,3,3)       (1,32,32,32)      46        74
9   Conv          FLOAT16   (1,7168,1,1),(43,7168,1,1)     (1,43,1,1)        133       617   ← Gemm tile0
12  Conv          FLOAT16   (1,7168,1,1),(43,7168,1,1)     (1,43,1,1)        133       617   ← Gemm tile1
13  ConvAdd       FLOAT16   (1,7168,1,1),(43,7168,1,1)     (1,43,1,1)        136       617   ← Gemm tile0
15  ConvAdd       FLOAT16   (1,7168,1,1),(43,7168,1,1)     (1,43,1,1)        189       617   ← Gemm tile1
18  ConvAdd       INT8      (1,4096,1,1),(43,4096,1,1)     (1,43,1,1)        84        178
```

- backbone 3 个 Conv：90 + 116 + 46 = **252 us（12%）**
- classifier 相关（Op 7~18）：≈ **900+ us（88%）**，其中 4 个 FLOAT16 Gemm 各搬 617KB

---

## 4. 根因分析（三层因果链）

```
① Generator 池化策略 (pool_every=2, 每隔一个 block 池化一次)
      ↓
② depth=3 时只触发 1 次池化 → 最终特征图 32×32×32（其他模型 pool 2 次 → 16×16）
      ↓
③ flatten 后 classifier 输入 32768 维（其他模型 16384）→ Linear 权重 43×32768 巨大
      ↓
④ RKNN 将大 Linear 转换为多个 FLOAT16 Gemm tile（量化后仍保留 fp16 精度）
      ↓
⑤ FLOAT16 在 NPU 上效率远低于 int8（NPU 原生加速 int8; fp16 走 DDR 搬运主导路径）
   每个 Gemm tile RW 617KB → 4 个 tile 共约 900us
```

### 佐证 1：训练精度同样受此缺陷影响

d3_k3 训练精度最低（94.38%）：32768 维 classifier 占参数比重过大 → 更易过拟合。
**同一结构缺陷同时伤害精度与速度。**

### 佐证 2：其他模型无此问题（对照）

cnn_test / d5_k3 / d5_k5 的最终特征图均为 16×16（pool 2 次），
classifier 输入 16384 维，Gemm 权重减半，实测延迟与 MACs 正相关，无异常。

---

## 5. 对课程各模块的意义

### 模块一（模型搜索）：搜索标准升级

不能只看"精度 + MACs"，还要检查**结构是否适配 NPU**：
最终特征图分辨率（决定 classifier 大小）是关键约束。
建议在 Generator 中增加约束：**最终特征图空间分辨率 ≤ 16×16**（或 classifier 输入 ≤ 16384 维），
从源头排除此类低效结构。

### 模块二（性能数学建模）：必须分层/分算子类型

简单模型 `delay = MACs / constant` 在此离群点上误差 >200%，无法满足"预判精度 ≥90%"。
正确形式：

```
delay ≈ Σ_conv (MACs / 高效算力) + Σ_gemm (参数 / 低效算力) + 固定开销
```

即：**按算子类型分别建模**（int8 Conv 与 FLOAT16 Gemm 分开），
本发现提供的关键实证数据 = 各算子类型的实际吞吐（见第 2 节表）。

### 模块三（模型分割）：分割点基于实测时间而非 MACs

d3_k3 的 backbone 仅 250us、classifier 900us——若在特征图处分割，
前端板 250us vs 后端板 900us，时延差 3.6 倍。
**分割算法必须使用逐层实测时间（eval_perf）作为输入，而非理论 MACs。**

### 模块四（自动化转化部署）：转换后检查算子分布

转 RKNN 后应检查是否存在大量 FLOAT16 Gemm（大 Linear 的标志）。
部署前可考虑用 GlobalAvgPool 替代大 Linear，或调整模型结构以减小 classifier。

---

## 6. 结论与建议

1. **保留该异常数据点**：它是"算子类型影响实际延迟"的真实实证，是模块二建模的关键校准数据。
2. **改进 Generator**：增加最终特征图分辨率约束（≤16×16），防止浅层模型产生超大 classifier。
3. **改进 Analyzer**：MACs 统计增加算子类型维度（Conv MACs / Gemm MACs 分开统计），
   使静态分析与实测延迟更可比。
4. **搜索空间调整**：在 README / 队友指南中标注此类结构不适合部署。

---

## 7. 复现方法

```bash
# 1. 转换模型（含 --verify 模拟器验证）
python convert_to_rknn.py --ckpt results/cnn_d3_c16-32-32_k3_best.pt \
    --depth 3 --channels 16,32,32 --kernel-size 3 \
    --out models/cnn_d3_k3.rknn --verify

# 2. 上板测速 + 精度
python verify_gtsrb.py --model models/cnn_d3_k3.rknn --num 2000 --bench 20

# 3. 逐层耗时（定位瓶颈）
python - <<'EOF'
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.load_rknn('models/cnn_d3_k3.rknn')
rknn.init_runtime(target='rk3566', perf_debug=True)
rknn.eval_perf()   # 观察 FLOAT16 Gemm 算子的 Time/RW
rknn.release()
EOF
```

---

*发现人: 课程小组 | 后续发现请按 FINDING-00X 编号继续归档于本目录*
