#ifndef FCRN_H_
#define FCRN_H_

#include "Model.h"
#include "Tensor.h"
#include <opencv2/opencv.hpp>

class FCRN
{
public:
    FCRN(const std::string pb_path);

    void inference(std::vector<cv::Mat> &input_batch, std::vector<cv::Mat> &output_batch);

    // Preprocess & Postprocess for image
    void preprocess(std::vector<cv::Mat> &img_batch);
    void postprocess(std::vector<cv::Mat> &img_batch);

private:
    // Convert between vector<Mat> <-> Tensor
    void setInputTensor(std::vector<cv::Mat> img_batch);
    std::vector<cv::Mat> getOutputTensor();

    // Original image information
    int batch_size_;
    cv::Size orig_size_;

    // Input & output size of model
    const cv::Size in_size_ = cv::Size(304, 228);
    const cv::Size out_size_ = cv::Size(160, 128);

    // Model & Placeholders
    std::shared_ptr<Model> model_ = nullptr;
    std::shared_ptr<Tensor> input_ = nullptr;
    std::shared_ptr<Tensor> output_ = nullptr;
};

#endif