# Models 说明文档

本目录所有 `.rknn` 模型的配置、来源与实测数据。实测平台: K11C / RK3566, NPU 900MHz, INT8, 输入 64×64×3。

| 文件 | 来源 | 配置 | 实测延迟 | 实测内存 | 精度 |
|------|------|------|---------|---------|------|
| `cnn_test.rknn` | **训练模型** | depth=4, ch=[16, 32, 64, 64], k=3, pool=auto(pool_every=2) | 0.89 ms | 0.95 MB | - |
| `cnn_d3_k3.rknn` | **训练模型** | depth=3, ch=[16, 32, 32], k=3, pool=auto(pool_every=2) | 1.61 ms | 2.76 MB | 94.4% |
| `cnn_d5_k3.rknn` | **训练模型** | depth=5, ch=[32, 32, 64, 64, 128], k=3, pool=auto(pool_every=2) | 1.79 ms | 2.95 MB | 97.7% |
| `cnn_d5_k5.rknn` | **训练模型** | depth=5, ch=[32, 32, 64, 64, 128], k=5, pool=auto(pool_every=2) | 2.46 ms | 3.18 MB | 97.9% |
| `exp_A1.rknn` | A: depth 实验 | depth=3, ch=[16, 32, 64], k=3, pool=[2, 3] | 0.78 ms | - | - |
| `exp_A2.rknn` | A: depth 实验 | depth=4, ch=[16, 32, 32, 64], k=3, pool=[2, 4] | 0.81 ms | - | - |
| `exp_A3.rknn` | A: depth 实验 | depth=5, ch=[16, 32, 32, 64, 64], k=3, pool=[2, 5] | 0.91 ms | - | - |
| `exp_A4.rknn` | A: depth 实验 | depth=6, ch=[16, 32, 32, 64, 64, 64], k=3, pool=[2, 6] | 0.99 ms | - | - |
| `exp_A5.rknn` | A: depth 实验 | depth=7, ch=[16, 32, 32, 64, 64, 64, 64], k=3, pool=[2, 7] | 1.15 ms | - | - |
| `exp_B1.rknn` | B: kernel 实验 | depth=4, ch=[32, 32, 64, 64], k=3-3-3-3, pool=[2, 4] | 0.97 ms | - | - |
| `exp_B2.rknn` | B: kernel 实验 | depth=4, ch=[32, 32, 64, 64], k=5-5-5-5, pool=[2, 4] | 1.45 ms | - | - |
| `exp_B3.rknn` | B: kernel 实验 | depth=4, ch=[32, 32, 64, 64], k=3-3-5-5, pool=[2, 4] | 1.21 ms | - | - |
| `exp_B4.rknn` | B: kernel 实验 | depth=4, ch=[32, 32, 64, 64], k=5-5-3-3, pool=[2, 4] | 1.28 ms | - | - |
| `exp_C1.rknn` | C: feature map 实验 | depth=4, ch=[32, 32, 64, 64], k=3, pool=[4] | 3.66 ms | - | - |
| `exp_C2.rknn` | C: feature map 实验 | depth=4, ch=[32, 32, 64, 64], k=3, pool=[2, 4] | 1.00 ms | - | - |
| `exp_C3.rknn` | C: feature map 实验 | depth=4, ch=[32, 32, 64, 64], k=3, pool=[2, 3, 4] | 0.61 ms | - | - |
| `exp_C4.rknn` | C: feature map 实验 | depth=4, ch=[32, 32, 64, 64], k=3, pool=[3, 4] | 1.24 ms | - | - |
| `exp_C5.rknn` | C: feature map 实验 | depth=4, ch=[32, 32, 64, 64], k=3, pool=[2, 3] | 0.91 ms | - | - |
| `exp_D1.rknn` | D: classifier 实验 | depth=4, ch=[32, 64, 64, 128], k=3, pool=[2, 3, 4] | 0.98 ms | - | - |
| `exp_D2.rknn` | D: classifier 实验 | depth=4, ch=[32, 32, 64, 64], k=3, pool=[2, 4] | 0.95 ms | - | - |
| `exp_D3.rknn` | D: classifier 实验 | depth=4, ch=[32, 64, 64, 128], k=3, pool=[2, 4] | 2.14 ms | - | - |
| `exp_D4.rknn` | D: classifier 实验 | depth=4, ch=[32, 32, 64, 64], k=3, pool=[4] | 3.52 ms | - | - |
| `val_V1.rknn` | 验证集(独立) | depth=3, ch=[32, 16, 64], k=3, pool=[2, 3] | 0.74 ms | - | - |
| `val_V2.rknn` | 验证集(独立) | depth=4, ch=[64, 32, 64, 64], k=3, pool=[2, 4] | 1.20 ms | - | - |
| `val_V3.rknn` | 验证集(独立) | depth=5, ch=[16, 32, 64, 64, 32], k=3, pool=[2, 4, 5] | 0.60 ms | - | - |
| `val_V4.rknn` | 验证集(独立) | depth=4, ch=[32, 32, 64, 128], k=5, pool=[2, 4] | 2.59 ms | - | - |
| `val_V5.rknn` | 验证集(独立) | depth=3, ch=[64, 64, 128], k=3, pool=[2, 3] | 2.40 ms | - | - |
| `val_V6.rknn` | 验证集(独立) | depth=5, ch=[32, 32, 64, 128, 128], k=3, pool=[3, 5] | 2.50 ms | - | - |
| `val_V7.rknn` | 验证集(独立) | depth=4, ch=[16, 32, 64, 128], k=3, pool=[2, 4] | 1.93 ms | - | - |
| `val_V8.rknn` | 验证集(独立) | depth=4, ch=[32, 64, 64, 128], k=3, pool=[2, 3] | 1.90 ms | - | - |
| `val_V9.rknn` | 验证集(独立) | depth=6, ch=[16, 32, 32, 64, 64, 128], k=3, pool=[2, 4, 6] | 0.73 ms | - | - |
| `val_V10.rknn` | 验证集(独立) | depth=5, ch=[32, 32, 64, 64, 128], k=5, pool=[2, 4, 5] | 1.42 ms | - | - |
| `val_V11.rknn` | 验证集(独立) | depth=3, ch=[32, 32, 64], k=3, pool=[3] | 3.19 ms | - | - |
| `val_V12.rknn` | 验证集(独立) | depth=4, ch=[64, 64, 64, 128], k=3, pool=[2, 4] | 2.39 ms | - | - |

---

## 分类说明

- **训练模型** (4 个): GTSRB 完整训练, 有真实精度, 用于精度/效率分析 (benchmark_results.csv 的 legacy 组)
- **控制变量实验** (exp_*, 随机权重): 第二阶段控制变量实验, 仅测 latency (与权重值无关), 分 A depth / B kernel / C feature map / D classifier 四组
- **验证集** (val_*, 随机权重): 第三阶段独立验证集, 用于验证 latency 模型泛化能力 (未参与拟合)
- **JSON 模型**: `rknn_latency_v1.json` (延迟回归参数) / `rknn_memory_v1.json` (内存校准参数)

> 精度 = 训练时 test accuracy (仅训练模型有); 随机权重模型无精度 (只测结构性能)。

