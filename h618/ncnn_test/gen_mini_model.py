"""
生成最小 NCNN 模型 (mini.param / mini.bin)
==========================================
网络: Input(1,3,64,64) → Conv(3→4,k3,p1) → ReLU → MaxPool(2) → Flatten → FC(→5)
用途: 验证 H618 上 NCNN CPU / Vulkan 推理链 (阶段 7), 非真实分类模型。

param 格式 (ncnn v2.x):
  Convolution: 0=num_out 1=kernel 2=dilation 3=stride 4=pad 5=bias_term 6=weight_size
  InnerProduct: 0=num_out 1=bias_term 2=weight_size

bin 格式: 按 param 权重层顺序, fp32 little-endian, 每层 weight 先 bias 后。
"""
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 网络结构 ----------
IN_C, IN_H, IN_W = 3, 64, 64
CONV_OUT = 4          # conv 输出通道
K = 3
POOL_OUT_H = IN_H // 2   # 32
FC_OUT = 5            # 输出维度

conv_w = CONV_OUT * IN_C * K * K      # 4*3*3*3 = 108
conv_b = CONV_OUT
fc_in = CONV_OUT * POOL_OUT_H * POOL_OUT_H   # 4*32*32 = 4096
fc_w = fc_in * FC_OUT
fc_b = FC_OUT

param = f"""7767517
7 6
Input            input   0 1 input 0={IN_C} 1={IN_H} 2={IN_W}
Convolution      conv1   1 1 input c1 0={CONV_OUT} 1={K} 2=1 3=1 4=1 5=1 6={conv_w}
ReLU             relu1   1 1 c1 r1
Pooling          pool1   1 1 r1 p1 0=0 1=3 2=2 3=2
Flatten          flat1   1 1 p1 f1
InnerProduct     fc1     1 1 f1 out 0={FC_OUT} 1=1 2={fc_w}
"""

# ---------- 权重 (固定可复现) ----------
rng = np.random.RandomState(42)
w_conv = (rng.rand(conv_w).astype(np.float32) - 0.5) * 0.1
b_conv = np.zeros(conv_b, dtype=np.float32)
w_fc = (rng.rand(fc_w).astype(np.float32) - 0.5) * 0.01
b_fc = np.zeros(fc_b, dtype=np.float32)

bin_data = np.concatenate([w_conv, b_conv, w_fc, b_fc])

with open(os.path.join(OUT_DIR, 'mini.param'), 'w') as f:
    f.write(param)
with open(os.path.join(OUT_DIR, 'mini.bin'), 'wb') as f:
    f.write(bin_data.tobytes())

print(f'生成 mini.param + mini.bin')
print(f'  结构: Input(1,{IN_C},{IN_H},{IN_W}) → Conv({CONV_OUT}) → ReLU '
      f'→ Pool → Flatten({fc_in}) → FC({FC_OUT})')
print(f'  bin: {len(bin_data)} floats = {len(bin_data)*4} bytes')
