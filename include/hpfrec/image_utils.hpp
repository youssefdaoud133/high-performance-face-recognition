#pragma once

#include <opencv2/opencv.hpp>
#include <wx/bitmap.h>

#include <string>

namespace hpfrec
{
    constexpr int kTrainingWidth = 100;
    constexpr int kTrainingHeight = 100;

    cv::Mat LoadPreparedImage(const std::string &path);
    cv::Mat LoadPreviewImage(const std::string &path);
    cv::Mat FlattenImage(const cv::Mat &image);
    wxBitmap MatToBitmap(const cv::Mat &input);
}
