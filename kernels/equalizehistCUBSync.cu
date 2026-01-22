__global__ void LUT(uint8_t* in, uint8_t* out, uint8_t* lut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = lut[in[i]];
    }
}

__global__ void buildLUT(uint32_t* cdf, uint8_t* lut, int n) {
    __shared__ uint32_t cdf_min;
    int i = threadIdx.x;

    if (i == 0) {
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

void histogramDeviceCUBSync(uint8_t* d_img, uint32_t* d_histogram, int N) {

    int      num_levels = 257; //bo 256 bin�w 0-255
    float    lower_level = 0.0;   //dolna granica
    float    upper_level = 256.0;   //g�rba granica

    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;


    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_img, d_histogram, num_levels,
        lower_level, upper_level, N);
    cudaDeviceSynchronize();

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Compute histograms
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_img, d_histogram, num_levels,
        lower_level, upper_level, N);
    cudaDeviceSynchronize();

    cudaFree(d_temp_storage);
}

void cdfDeviceCUBSync(uint32_t* d_histogram, uint32_t* d_cdf) {
    int  num_items = 256;

    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_histogram, d_cdf, num_items);
    cudaDeviceSynchronize();

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_histogram, d_cdf, num_items);
    cudaDeviceSynchronize();

    cudaFree(d_temp_storage);
}

void equalizeHistDeviceCUBSync(cv::Mat& img, cv::Mat& res) {

    int width = img.cols;
    int height = img.rows;
    int N = width * height;

    int blockDim = 256;
    int gridDim = (N + 255) / 256;

    uint8_t* d_img = nullptr;
    uint8_t* d_res = nullptr;
    uint32_t* d_histogram = nullptr;
    uint8_t* d_lut = nullptr;
    uint32_t* d_cdf = nullptr;

    cudaMalloc(&d_img, N * sizeof(uint8_t));
    cudaMalloc(&d_res, N * sizeof(uint8_t));
    cudaMalloc(&d_histogram, 256 * sizeof(uint32_t));
    cudaMalloc(&d_lut, 256 * sizeof(uint8_t));
    cudaMalloc(&d_cdf, 256 * sizeof(uint32_t));

    cudaMemcpy(d_img, img.data, N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    histogramDeviceCUBSync(d_img, d_histogram, N);
    cudaDeviceSynchronize();
    cdfDeviceCUBSync(d_histogram, d_cdf);
    cudaDeviceSynchronize();

    buildLUT << <1, 256 >> > (d_cdf, d_lut, N);
    cudaDeviceSynchronize();
    LUT << <gridDim, blockDim >> > (d_img, d_res, d_lut, N);
    cudaDeviceSynchronize();

    cudaMemcpy(res.data, d_res, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_res);
    cudaFree(d_histogram);
    cudaFree(d_lut);
    cudaFree(d_cdf);

}
