#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include "utils.cuh"


__global__  void LUT_Thrust(uint8_t* in, uint8_t* out, uint8_t* lut, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        out[idx] = lut[in[idx]];
    }
}

__global__ void buildLUT_Thrust(uint32_t* cdf, uint8_t* lut, int n) {
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


template <typename Vector1, typename Vector2>
void histogramDeviceThrust(Vector1& input, Vector2& histogram) {

    typedef typename Vector1::value_type ValueType;
    typedef typename Vector2::value_type IndexType;

    thrust::device_vector<ValueType> data(input);
    thrust::sort(thrust::device,data.begin(), data.end());

    IndexType num_bins = 256;

    histogram.resize(num_bins);

    thrust::counting_iterator<IndexType> search_begin(0);
    thrust::upper_bound(thrust::device,data.begin(), data.end(),
        search_begin, search_begin + num_bins,
        histogram.begin());

    thrust::adjacent_difference(thrust::device,histogram.begin(), histogram.end(), histogram.begin());
}

torch::Tensor equalizeHistThrust(torch::Tensor img) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    assert(img.dim() == 2);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const int N = width * height;

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width},torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    thrust::device_vector<uint8_t> d_img(img.data_ptr<uint8_t>(), img.data_ptr<uint8_t>() + N);
    thrust::device_vector<uint32_t> d_cdf;
    thrust::device_vector<uint32_t> d_histogram;
    thrust::device_vector<uint8_t> d_lut(256);

    histogramDeviceThrust(d_img, d_histogram);
    cudaDeviceSynchronize();
    d_cdf.resize(d_histogram.size());

    thrust::inclusive_scan(thrust::device,d_histogram.begin(), d_histogram.end(), d_cdf.begin());
    cudaDeviceSynchronize();

    uint8_t* img_ptr = thrust::raw_pointer_cast(d_img.data());
    uint8_t* lut_ptr = thrust::raw_pointer_cast(d_lut.data());
    uint32_t* cdf_ptr = thrust::raw_pointer_cast(d_cdf.data());

    buildLUT_Thrust<<<1, 256>>>(cdf_ptr, lut_ptr,N);
    cudaDeviceSynchronize();
    LUT_Thrust<<<dimGrid, dimBlock>>>(img_ptr,result.data_ptr<uint8_t>(), lut_ptr,width,height);
    cudaDeviceSynchronize();

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
