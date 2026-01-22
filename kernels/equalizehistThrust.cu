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

template <typename Vector1, typename Vector2>
void histogramDeviceThrust(Vector1& input, Vector2& histogram) {

    typedef typename Vector1::value_type ValueType;
    typedef typename Vector2::value_type IndexType;

    thrust::device_vector<ValueType> data(input);
    thrust::sort(data.begin(), data.end());

    IndexType num_bins = 256;

    // resize histogram storage
    histogram.resize(num_bins);

    // find the end of each bin of values
    thrust::counting_iterator<IndexType> search_begin(0);
    thrust::upper_bound(data.begin(), data.end(),
        search_begin, search_begin + num_bins,
        histogram.begin());


    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());

}

void equalizeHistDeviceThrust(cv::Mat& img, cv::Mat& res) {
    int N = img.cols * img.rows;
    int blockDim = 256;
    int gridDim = (N + 255) / 256;

    thrust::device_vector<uint8_t> d_img(img.data, img.data + N);
    thrust::device_vector<uint8_t> d_out(d_img.size());
    thrust::device_vector<uint32_t> d_histogram;
    thrust::device_vector<uint32_t> d_cdf;
    thrust::device_vector<uint8_t> d_lut(256);

    histogramDeviceThrust(d_img, d_histogram);
    d_cdf.resize(d_histogram.size());



    thrust::inclusive_scan(d_histogram.begin(), d_histogram.end(), d_cdf.begin());

    uint8_t* img_ptr = thrust::raw_pointer_cast(d_img.data());
    uint8_t* out_ptr = thrust::raw_pointer_cast(d_out.data());
    uint8_t* lut_ptr = thrust::raw_pointer_cast(d_lut.data());
    uint32_t* cdf_ptr = thrust::raw_pointer_cast(d_cdf.data());

    buildLUT << <1, 256 >> > (cdf_ptr, lut_ptr, N);
    LUT << <gridDim, blockDim >> > (img_ptr, out_ptr, lut_ptr, N);

    cudaMemcpy(res.data, out_ptr, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}