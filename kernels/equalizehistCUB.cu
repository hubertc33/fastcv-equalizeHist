#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include "utils.cuh"

__global__  void LUT_CUB(uint8_t* in, uint8_t* out, uint8_t* lut, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        out[idx] = lut[in[idx]];
    }
}

__global__ void buildLUT_CUB(uint32_t* cdf, uint8_t* lut, int n) {
    __shared__ uint32_t cdf_min;
    int i = threadIdx.x;

    if (i == 0) {
        cdf_min=0;
        for (int j = 0; j < 256; j++) {
            if (cdf[j] != 0) {
                cdf_min = cdf[j];
                break;
            }
        }
    }
    __syncthreads();
    lut[i] = (uint8_t)roundf((float)(cdf[i] - cdf_min) / (n - cdf_min) * 255.0);
}


void histogramCUB(uint8_t* d_img, uint32_t* d_histogram, int N, cudaStream_t stream) {

    int      num_levels = 257;
    float    lower_level = 0.0;
    float    upper_level = 256.0;

    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;


    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_img, d_histogram, num_levels,
        lower_level, upper_level, N,stream);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_img, d_histogram, num_levels,
        lower_level, upper_level, N,stream);

    cudaFree(d_temp_storage);
}

void cdfCUB(uint32_t* d_histogram, uint32_t* d_cdf, cudaStream_t stream) {
    int  num_items = 256;

    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_histogram, d_cdf, num_items,stream);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_histogram, d_cdf, num_items,stream);

    cudaFree(d_temp_storage);
}


torch::Tensor equalizeHistCUB(torch::Tensor img) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    assert(img.dim() == 2);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const int N = width * height;

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto stream = at::cuda::getCurrentCUDAStream();

    auto result = torch::empty({height, width},torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    auto hist = torch::zeros({256}, torch::dtype(torch::kInt32).device(img.device()));
    auto cdf = torch::empty({256}, torch::dtype(torch::kInt32).device(img.device()));
    auto lut = torch::empty({256}, torch::dtype(torch::kByte).device(img.device()));

    histogramCUB(img.data_ptr<uint8_t>(), (uint32_t*)hist.data_ptr<int32_t>(), N, stream);
    cdfCUB((uint32_t*)hist.data_ptr<int32_t>(), (uint32_t*)cdf.data_ptr<int32_t>(), stream);

    buildLUT_CUB<<<1, 256, 0, stream>>>((uint32_t*)cdf.data_ptr<int32_t>(), lut.data_ptr<uint8_t>(),N);
    LUT_CUB<<<dimGrid, dimBlock, 0, stream>>>(img.data_ptr<uint8_t>(), result.data_ptr<uint8_t>(), lut.data_ptr<uint8_t>(),width,height);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
