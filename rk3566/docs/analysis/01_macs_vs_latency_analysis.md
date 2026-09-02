# RK3566 硬件感知分析：MACs 与实际推理时延

| 项 | 值 |
|----|----|
| 编号 | ANALYSIS-001 |
| 日期 | 2026-09-01 |
| 平台 | K11C / RK3566, librknnrt 1.5.2, NPU 900MHz |
| 数据 | `results/benchmark_results.csv`, `results/layer_latency.csv` |
| 图 | `docs/figures/fig1234_metrics_vs_latency.png`, `fig5_layer_latency_breakdown.png` |
| 关联 | FINDING-001 (FLOAT16 Gemm Classifier 瓶颈) |

---

## 1. 数据（4 个已有模型，统一 Benchmark 方法）

| 模型 | Params | Total MACs | Conv MACs | Linear MACs | Feature Map | Flatten | NPU Latency |
|------|-------:|-----------:|----------:|------------:|-------------|--------:|------------:|
| d3_k3 | 1,423,483 | 31.5M | 30.1M | 1.4M | 32×32×32 | 32768 | 1.594 ms |
| cnn_test | 765,243 | 78.0M | 77.3M | 0.7M | 64×16×16 | 16384 | 0.914 ms |
| d5_k3 | 1,548,811 | 118.2M | 116.8M | 1.4M | 128×16×16 | 32768 | 1.807 ms |
| d5_k5 | 1,796,107 | 325.8M | 324.4M | 1.4M | 128×16×16 | 32768 | 2.399 ms |

测量方法: warm-up 5 次 + 正式 50 次取平均, eval_perf 取 NPU 纯计算时间。
逐层耗时来自 `perf_debug=True` 的 eval_perf 输出。

## 2. 回归结果

| 模型 | 公式 | R² | MAE | MAPE |
|------|------|----:|----:|-----:|
| M1 总 MACs | T = 0.0037·MACs + 1.165 | **0.617** | 0.270 ms | **22.8%** |
| M2 分算子 | T = 0.0028·Conv + 1.13·Linear − 0.098 | **0.9997** | 0.008 ms | **0.4%** |
| M3 +flatten | T = 0.0028·MACs + 0.049·flattenK − 0.098 | **0.9997** | 0.008 ms | **0.4%** |

**警示 (Task 第 15 节)**: 仅 4 个数据点, M2/M3 拟合 3 参数(3 自由度), R² 高含过拟合成分,
只能作为**方向性证据**, 不构成最终性能模型。需更多数据点验证。

**系数解读**（方向性）:
- conv 系数 0.0028 ms/M ≈ 0.36 TOPS 有效算力
- linear 系数 1.13 ms/M ≈ **每单位 linear MACs 的开销是 conv 的 ~400 倍**
  （FLOAT16 Gemm + DDR 搬运主导, 见 FINDING-001）
- flatten 系数 0.049 ms/K: 每 1000 维 flatten 增加 ~0.05ms

## 3. 假设验证

| 假设 | 结论 | 证据 |
|------|------|------|
| H1: MACs 高 ≠ latency 一定高 | ✅ 支持 | d3_k3 (31.5M) 1.59ms > cnn_test (78M) 0.91ms |
| H2: 大 Feature Map → 高 latency | ✅ 支持 | FM 32×32 (d3_k3) 最慢; flatten 维度与延迟正相关 |
| H3: 大 Linear/GEMM classifier 是主要来源 | ✅ 支持 | d3_k3 classifier 占 88% 延迟 (900us/1020us) |
| H4: 相近 MACs 不同结构延迟不同 | ✅ 支持 | d3_k3 vs cnn_test 结构不同 → 延迟方向与 MACs 相反 |

## 4. 问题回答 (Task 第 19 节)

**Q1: MACs 能否很好预测 RK3566 latency?**
不能。M1 R²=0.617, MAPE=22.8%。d3_k3 是反例。但排除 classifier 异常结构后,
大范围内 (78M→326M) MACs 与延迟正相关。

**Q2: Params 能否预测?**
更差。Params 与延迟无明显单调关系 (d3_k3 参数最多却最慢, cnn_test 最少却最快)。

**Q3: Classifier 大小是否明显影响 latency?**
是。flatten 32768 的 classifier 固定开销 ~900us (FLOAT16 Gemm tile 化),
16384 的 ~400us。classifier 是 4 个模型中 3 个的最大单点开销来源。

**Q4: Feature Map 尺寸是否比 MACs 更有解释力?**
在"最终特征图决定 flatten 维度"的意义上, 是。FM 空间过大 (32×32) 或通道过多 (128)
都导致 flatten 32768 → classifier 瓶颈。特征图尺寸 → flatten → Gemm 开销,
是比总 MACs 更直接的延迟解释路径。

**Q5: 区分 Conv 和 Linear/GEMM 后预测是否改善?**
方向性改善明显 (M2 R² 0.617→0.9997), 但受 4 点过拟合影响, 结论限定为
"区分算子类型是正确方向, 需更多数据确认"。

**Q6: 能否根据指标提前过滤不适合的 CNN?**
能。`max_classifier_input_dim=20000` 静态过滤已实现并验证
(过滤 8/10 现有候选, 保留 flatten ≤16384 的安全结构)。

## 5. 结论与限制

**结论** (限定于当前测试模型 + RK3566 环境):
1. 理论总 MACs 不能可靠预测 RK3566 延迟 (R²=0.617)。
2. 算子类型是延迟的主要调节因素: int8 Conv 高效 (~0.36 TOPS),
   FLOAT16 GEMM 极低效 (~400× 每 MAC 开销)。
3. Classifier (flatten 维度) 是小型 CNN 的隐藏延迟源,
   由最终特征图尺寸决定, 是硬件感知过滤的有效指标。
4. 结构感知建模 (区分 Conv/Linear) 方向正确, 待更多数据验证。

**限制**:
- 仅 4 个模型, 统计功效低; 回归 R² 高含过拟合。
- perf_debug 模式本身会降低性能, 逐层数据用于相对分析。
- 未覆盖更深/更宽/不同池化策略的模型。
- 未建立可发布的性能预测模型 (未达 90% 精度声称标准)。

## 6. 复现

```bash
# 统一 benchmark (需板子连接)
python benchmark_rknn3566.py --num 50 --warmup 5 --layers

# 回归 + 图
python analyze_performance.py
```
