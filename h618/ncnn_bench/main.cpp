#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <ncnn/net.h>

// H618 NCNN Benchmark (Phase 1-3)
// 用法: ./h618_ncnn_bench model.param model.bin [cpu|vulkan] [iterations]
//   warmup 默认 20, iterations 默认 200
// 输入固定 64x64x3 (GTSRB), 输出 mean/median/min/max/std

static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "用法: h618_ncnn_bench model.param model.bin [cpu|vulkan] [iterations]" << std::endl;
        return 1;
    }
    std::string param_path = argv[1];
    std::string bin_path = argv[2];
    bool vulkan = (argc > 3 && std::string(argv[3]) == "vulkan");
    int iterations = (argc > 4) ? std::atoi(argv[4]) : 200;
    int warmup = 20;

    std::cout << "=== H618 NCNN Bench ===" << std::endl;
    std::cout << "  model: " << param_path << " + " << bin_path << std::endl;
    std::cout << "  backend: " << (vulkan ? "VULKAN" : "CPU")
              << " | warmup=" << warmup << " iterations=" << iterations
              << std::endl;

    if (vulkan)
    {
        ncnn::create_gpu_instance();
        std::cout << "  Vulkan GPU count: " << ncnn::get_gpu_count() << std::endl;
    }

    double mean = 0, median = 0, stddev = 0, p95 = 0, tmin = 0, tmax = 0;
    {
        ncnn::Net net;   // 内层作用域: 先于 destroy_gpu_instance 析构
        net.opt.num_threads = 4;
        net.opt.use_vulkan_compute = vulkan;

        if (net.load_param(param_path.c_str()) != 0 ||
            net.load_model(bin_path.c_str()) != 0)
        {
            std::cerr << "load model failed" << std::endl;
            if (vulkan) ncnn::destroy_gpu_instance();
            return 1;
        }

        // 输入 1x3x64x64 (GTSRB 尺寸), 固定值
        ncnn::Mat in(64, 64, 3);
        in.fill(0.5f);

        // warm-up (排除首次初始化)
        for (int i = 0; i < warmup; i++)
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ncnn::Mat out;
            ex.extract("out0", out);
        }

        // 正式测量
        std::vector<double> times;
        times.reserve(iterations);
        for (int i = 0; i < iterations; i++)
        {
            double t0 = now_ms();
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ncnn::Mat out;
            ex.extract("out0", out);
            times.push_back(now_ms() - t0);
        }

        // 统计
        std::sort(times.begin(), times.end());
        double sum = 0;
        for (double t : times) sum += t;
        mean = sum / times.size();
        median = times[times.size() / 2];
        double var = 0;
        for (double t : times) var += (t - mean) * (t - mean);
        stddev = std::sqrt(var / times.size());
        p95 = times[(int)(times.size() * 0.95)];
        tmin = times.front();
        tmax = times.back();
    }   // net 在此析构 (先于 destroy_gpu_instance)

    std::cout << "--- results ---" << std::endl;
    std::cout << "  mean_ms=" << mean << std::endl;
    std::cout << "  median_ms=" << median << std::endl;
    std::cout << "  min_ms=" << tmin << std::endl;
    std::cout << "  max_ms=" << tmax << std::endl;
    std::cout << "  std_ms=" << stddev << std::endl;
    std::cout << "  p95_ms=" << p95 << std::endl;

    // CSV 行 (便于收集)
    std::cout << "CSV: " << param_path << "," << (vulkan ? "vulkan" : "cpu")
              << "," << mean << "," << median << "," << tmin
              << "," << tmax << "," << stddev << "," << p95
              << std::endl;

    if (vulkan) ncnn::destroy_gpu_instance();
    return 0;
}
