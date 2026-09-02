#!/bin/bash
# H618 NCNN benchmark 批量收集 (CPU + Vulkan)
# 用法: ./run_h618_bench.sh "model1 model2 ..." [iterations]
set -e
H618BENCH=/data/local/tmp/h618bench
MODELS="$1"
ITERS="${2:-200}"
OUT=results/h618_latency.csv
mkdir -p results

echo "model,backend,mean_ms,median_ms,min_ms,max_ms,std_ms,p95_ms" > $OUT

for m in $MODELS; do
    for backend in cpu vulkan; do
        echo "== $m $backend =="
        adb shell "cd $H618BENCH && ./h618_ncnn_bench models/$m.param models/$m.bin $backend $ITERS" 2>/dev/null \
            | grep "^CSV:" | sed "s/^CSV: //" | sed "s|\.param||" >> $OUT
    done
done
echo "===== 完成: $OUT ====="
