# Next Steps（下一步：已回答 vs 待回答）

当前阶段收口后的问题清单，避免以后重新思考"要做什么"。

## 已经回答的问题（有实验支撑）

| Q | 结论 |
|---|------|
| Q1 MACs 能解释 RK3566 latency？ | 不能单独解释（M1 R²=0.55），需算子类型（M2 R²=0.99） |
| Q1' MACs 能解释 H618 CPU latency？ | **基本可以**（相关性 0.99），但 Conv+Linear 更稳（CV MAPE 6.5%） |
| Q2 RK3566 与 H618 对算子表现不同？ | 是：RK linear/conv=325× vs H618 13× |
| Q3 Flatten-heavy 使 H618 获绝对优势？ | **当前没有**（RK 绝对延迟仍全胜） |
| Q4 单模型异构切分降 latency？ | **当前无证据**（T_hetero > T_RK 必然） |
| Q5 RK3566 内存可预测？ | 可（校准后 MAPE 2.3~2.6%） |

## 尚未回答的问题（待后续阶段）

| Q | 需要的输入 |
|---|-----------|
| Q6 Accuracy 与 hardware cost 如何共同优化？ | 队友 accuracy 数据 |
| Q7 是否存在 Pareto-optimal architecture？ | accuracy + latency/memory 联合分析 |
| Q8 通信成本加入后是否有 partition point？ | 真实通信 benchmark |
| Q9 异构执行能否提升 throughput？ | 流水线/并发设计 + 实测 |
| Q10 多模型并发时 RK3566+H618 资源互补？ | 调度实验 |
| Q11 Vulkan 在更大模型/不同 workload 是否有价值？ | 更大输入/模型测试 |

## 推荐执行顺序

```
① 等队友 accuracy 数据 (model_id → accuracy)
     ↓
② Hardware-aware Pareto Search:
   evaluate_candidate (latency/memory/params 已就绪) + accuracy
   → Pareto Front → 最优模型 (RK3566 部署)
     ↓
③ (可选) 异构方向:
   若目标吞吐/多模型并发 → 真实通信 benchmark (FM 16KB~64KB)
   → 判断并发切分是否值得
     ↓
④ 最终系统验证 + 课程报告整理
```

## 决策依据

- 若目标是**课程交付**（单板最优模型）：做 ①② 即可，异构非必需
- 若目标是**异构演示**：③ 需要通信硬件与工程投入，先确认资源
- 当前实验表明：异构的卖点是"并发/资源互补"，不是"单模型更快"
