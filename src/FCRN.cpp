#include "FCRN.h"

FCRN::FCRN(const std::string pb_path)
{
    // Restore model
    model_ = new Model(pb_path);

    // Prepare placeholder for model
    input_ = new Tensor(*model_, "input_image");
    output_ = new Tensor(*model_, "ConvPred/ConvPred");
}

void FCRN::inference(std::vector<cv::Mat> &input_batch,
                     std::vector<cv::Mat> &output_batch)
{
    // Original image size (all the images in batch should be same size)
    batch_size_ = int(input_batch.size());
    orig_size_ = input_batch[0].size();

    // Preprocess
    this->pre_process(input_batch);

    // Convert cv::Mat to tensor
    this->set_input_tensor(input_batch);

    // Run model
    model_->run(*input_, *output_);

    // Convert tensor to cv::Mat
    output_batch = this->get_output_tensor();

    // Postprocess
    this->post_process(output_batch);
}

void FCRN::pre_process(std::vector<cv::Mat> &img_batch)
{
    for (int i = 0; i < batch_size_; i++)
    {
        cv::resize(img_batch[i], img_batch[i], in_size_); // CV_8UC1 (is it OK...?)
    };
}

void FCRN::post_process(std::vector<cv::Mat> &out_batch)
{
    for (int i = 0; i < out_batch.size(); i++)
    {
        cv::resize(out_batch[i], out_batch[i], orig_size_);
    };
}

void FCRN::set_input_tensor(std::vector<cv::Mat> img_batch)
{
    std::vector<float> batch_data;
    for (int i = 0; i < batch_size_; i++)
    {
        // Convert to vector format
        std::vector<float> img_data;
        img_data.assign(img_batch[i].data, img_batch[i].data + img_batch[i].total() * img_batch[i].channels());
        // Concat
        batch_data.insert(batch_data.end(), img_data.begin(), img_data.end());
    }
    // Convert to tensor
    input_->set_data<float>(batch_data, {batch_size_, in_size_.height, in_size_.width, 3});
}

std::vector<cv::Mat> FCRN::get_output_tensor()
{
    // Extract prediction result as vector format
    std::vector<float> tensor_vec = output_->get_data<float>();
    int tensor_size = out_size_.height * out_size_.width * 1;

    // Convert each vector to cv::Mat format
    std::vector<cv::Mat> out_batch;
    for (int i = 0; i < batch_size_; i++)
    {
        // Extract ith_data from tensor_vec
        std::vector<float> ith_vec(tensor_size);
        copy(tensor_vec.begin() + i * tensor_size,
             tensor_vec.begin() + (i + 1) * tensor_size,
             ith_vec.begin());

        // Convert std::vec<float> to cv::Mat
        // - [Ref] https://answers.opencv.org/question/81831/convert-stdvectordouble-to-mat-and-show-the-image/
        cv::Mat ith_mat = cv::Mat(out_size_.height, out_size_.width, CV_32FC1);
        memcpy(ith_mat.data, ith_vec.data(), ith_vec.size() * sizeof(float));
        out_batch.push_back(ith_mat);
    }

    return out_batch;
}
