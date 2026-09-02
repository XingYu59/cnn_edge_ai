#include <iostream>
#include <cmath>
#include <ncnn/net.h>

// 阶段 7: 用最小模型 (mini.param/bin) 验证 CPU 与 Vulkan 推理链。
// 用法: ./infer_mini (需与 mini.param/mini.bin 同目录)
static float run_inference(ncnn::Net& net, bool vulkan)
{
    net.opt.use_vulkan_compute = vulkan;
    net.opt.num_threads = 4;

    if (net.load_param("mini.param") != 0 ||
        net.load_model("mini.bin") != 0)
    {
        std::cerr << "load model failed" << std::endl;
        return -1.0f;
    }

    // 输入 1x3x64x64, 固定值
    ncnn::Mat in(64, 64, 3);
    in.fill(0.5f);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret != 0)
    {
        std::cerr << "extract failed, ret=" << ret << std::endl;
        return -1.0f;
    }

    // 输出 1x1x5 (FC 输出); 注意 ncnn Mat 每行有 align padding
    std::cout << "  output: w=" << out.w << " h=" << out.h
              << " c=" << out.c << std::endl;
    float* ptr = (float*)out.data;
    int valid = out.w * out.h * out.c;   // 本例 c=1, 连续 5 个有效
    float sum = 0.0f;
    for (int i = 0; i < valid; i++)
        sum += ptr[i] * ptr[i];
    float norm = std::sqrt(sum);
    std::cout << "  output norm = " << norm << std::endl;
    return norm;
}

int main()
{
    std::cout << "=== Mini Model Inference Test ===" << std::endl;

    // CPU 推理
    std::cout << "-- CPU inference --" << std::endl;
    ncnn::Net cpu_net;
    float cpu_norm = run_inference(cpu_net, false);

    // Vulkan 推理 (net 需在 destroy_gpu_instance 前析构)
    std::cout << "-- Vulkan inference --" << std::endl;
    ncnn::create_gpu_instance();
    float vk_norm = -1.0f;
    {
        ncnn::Net vk_net;
        vk_norm = run_inference(vk_net, true);
    }   // vk_net 在此析构 (先于 GPU 实例销毁)
    ncnn::destroy_gpu_instance();

    if (cpu_norm > 0 && vk_norm > 0)
    {
        float diff = std::fabs(cpu_norm - vk_norm);
        std::cout << "=== CPU norm=" << cpu_norm
                  << " vs Vulkan norm=" << vk_norm
                  << " diff=" << diff << std::endl;
        if (diff < 1e-3f)
            std::cout << "CPU/Vulkan results consistent. OK." << std::endl;
        else
            std::cout << "results differ (norm diff=" << diff << ")" << std::endl;
    }
    else
    {
        std::cout << "one or both inference failed" << std::endl;
    }
    return 0;
}
