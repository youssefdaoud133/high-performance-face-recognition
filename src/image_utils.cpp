#include "hpfrec/image_utils.hpp"

#include <cstring>

namespace hpfrec
{
    cv::Mat LoadPreparedImage(const std::string &path)
    {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            return cv::Mat();
        }

        cv::resize(image, image, cv::Size(kTrainingWidth, kTrainingHeight));
        cv::equalizeHist(image, image);
        image.convertTo(image, CV_32F, 1.0 / 255.0);
        return image;
    }

    cv::Mat FlattenImage(const cv::Mat &image)
    {
        return image.reshape(1, static_cast<int>(image.total())).clone();
    }

    wxBitmap MatToBitmap(const cv::Mat &input)
    {
        cv::Mat display;
        if (input.empty())
        {
            display = cv::Mat(kTrainingHeight, kTrainingWidth, CV_8UC3, cv::Scalar(32, 32, 32));
        }
        else if (input.channels() == 3)
        {
            if (input.type() == CV_8UC3)
            {
                display = input;
            }
            else
            {
                input.convertTo(display, CV_8UC3);
            }
        }
        else
        {
            cv::Mat gray8;
            if (input.depth() == CV_32F || input.depth() == CV_64F)
            {
                cv::Mat normalized;
                cv::normalize(input, normalized, 0, 255, cv::NORM_MINMAX);
                normalized.convertTo(gray8, CV_8U);
            }
            else
            {
                input.convertTo(gray8, CV_8U);
            }
            cv::cvtColor(gray8, display, cv::COLOR_GRAY2BGR);
        }

        cv::Mat rgb;
        cv::cvtColor(display, rgb, cv::COLOR_BGR2RGB);
        wxImage image(rgb.cols, rgb.rows);
        std::memcpy(image.GetData(), rgb.data, static_cast<size_t>(rgb.total()) * 3);
        return wxBitmap(image);
    }
}
