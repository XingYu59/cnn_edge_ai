"""
hpm: Hardware Performance Model 包 (RK3566)
===========================================
把已完成的 RK3566 实验成果封装为可调用接口:
  architecture.py  - 静态架构分析 (不依赖实机)
  latency.py       - RK3566 latency 预测模型 (拟合/加载/预测)
  memory.py        - 内存估算
  filter.py        - 硬件约束过滤
  pipeline.py      - evaluate_candidate 最小 pipeline
"""
