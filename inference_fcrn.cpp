#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "include/FCRN.h"

std::string pb_file = "../freeze_graph/pb_file/NYU_FCRN.pb";
std::string img_dir = "../samples/home_office_0013";

void parseArguments(char *arg)
{
    char buf[1000];
    if (1 == sscanf(arg, "pb_file=%s", buf))
    {
        pb_file = buf;
        return;
    }
    if (1 == sscanf(arg, "img_dir=%s", buf))
    {
        img_dir = buf;
        return;
    }
}

int getImgsFromDir(std::string dir, std::vector<std::string> &files)
{
    // Get image files (png format) in specified directory
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }
    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);
        if (name != "." && name != ".." && name.substr(name.size() - 3, name.size()) == "png")
            files.push_back(name);
    }
    closedir(dp);

    // Sort images
    std::sort(files.begin(), files.end());
    if (dir.at(dir.length() - 1) != '/')
        dir = dir + "/";
    for (unsigned int i = 0; i < files.size(); i++)
    {
        if (files[i].at(0) != '/')
            files[i] = dir + files[i];
    }
    std::cout << "Num of images: " << files.size() << std::endl;

    return files.size();
}

int main(int argc, char **argv)
{
    // Arguments
    for (int i = 1; i < argc; i++)
        parseArguments(argv[i]);

    // Construct model by specified pb-file
    FCRN model(pb_file);

    // Prepare imgs from specified directory
    std::vector<std::string> img_files;
    int num_imgs = getImgsFromDir(img_dir, img_files);
    for (int i = 0; i < num_imgs; i++)
    {
        // Read image
        std::vector<cv::Mat> img_batch;
        cv::Mat img = cv::imread(img_files[i]);
        img_batch.push_back(img);
        std::cout << i << "/" << num_imgs << " : " << img_files[i] << std::endl;

        // Inference
        std::vector<cv::Mat> depth_batch;
        model.inference(img_batch, depth_batch);

        // Normalize for visualization (colormap)
        double min_val, max_val;
        cv::minMaxLoc(depth_batch[0], &min_val, &max_val);
        cv::Mat depth_viz;
        depth_viz = 255 * (depth_batch[0] - min_val) / (max_val - min_val);
        depth_viz.convertTo(depth_viz, CV_8UC1);
        cv::applyColorMap(depth_viz, depth_viz, 2);

        // Concat image & its prediction
        cv::Mat cat_result(cv::Size(img.cols * 2, img.rows), img.type(), cv::Scalar::all(0));
        cv::Mat mat_roi = cat_result(cv::Rect(0, 0, img.cols, img.rows));
        img.copyTo(mat_roi); // copy to left side of cat_result
        mat_roi = cat_result(cv::Rect(img.cols, 0, img.cols, img.rows));
        depth_viz.copyTo(mat_roi); // copy to right side of cat_result

        // Visualize
        cv::namedWindow("FULL", cv::WINDOW_AUTOSIZE);
        cv::imshow("FULL", cat_result);
        cv::waitKey(1);
    }
}
