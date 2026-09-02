#include <iostream>
#include <ncnn/net.h>

int main()
{
    std::cout << "Hello NCNN!" << std::endl;

    ncnn::Net net;
    std::cout << "NCNN Net created successfully." << std::endl;

    // ---- Vulkan 能力检查 (阶段 6) ----
    ncnn::create_gpu_instance();

    int gpu_count = ncnn::get_gpu_count();
    std::cout << "Vulkan GPU count: " << gpu_count << std::endl;

    if (gpu_count > 0)
    {
        for (int i = 0; i < gpu_count; i++)
        {
            const ncnn::GpuInfo& gi = ncnn::get_gpu_info(i);
            std::cout << "  GPU[" << i << "]: " << gi.device_name()
                      << std::endl;
        }
        std::cout << "Vulkan backend available." << std::endl;
    }
    else
    {
        std::cout << "Vulkan backend NOT available." << std::endl;
    }

    ncnn::destroy_gpu_instance();

    std::cout << "Done." << std::endl;
    return 0;
}
