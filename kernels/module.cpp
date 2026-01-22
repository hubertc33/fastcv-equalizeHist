#include <torch/extension.h>

// forward declarations
torch::Tensor rgb_to_gray(torch::Tensor img);
torch::Tensor box_blur(torch::Tensor img, int blurSize);
torch::Tensor sobel(torch::Tensor img);
torch::Tensor dilation(torch::Tensor img, int filterSize);
torch::Tensor erosion(torch::Tensor img, int filterSize);
torch::Tensor equalizeHistCUB(torch::Tensor img);
torch::Tensor equalizeHistCUBSync(torch::Tensor img);
//void equalizeHistDeviceThrust(cv::Mat& img, cv::Mat& res);
//void equalizeHistDeviceThrustAsync(cv::Mat& img, cv::Mat& res);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rgb2gray", &rgb_to_gray, "rgb to grayscale kernel");
    m.def("blur", &box_blur, "box blur kernel");
    m.def("sobel", &sobel, "sobel filter kernel");
    m.def("dilate", &dilation, "dilation kernel");
    m.def("erode", &erosion, "erosion kernel");
    m.def("equalizeHistCUB", &equalizeHistCUB, "equalizehist kernel using CUB");
    //m.def("equalizehistThrust", &equalizeHistThrust, "equalizehist kernel using Thrust");
    m.def("equalizeHistCUBSync", &equalizeHistCUBSync, "equalizehist kernel using CUB synchronic version");
    //m.def("equalizehistThrustAsync", &equalizeHistThrustAsync, "equalizehist kernel using Thrust asynchronic version");
}