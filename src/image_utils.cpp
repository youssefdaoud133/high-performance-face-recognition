#include "hpfrec/image_utils.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <cstring>
#include <mutex>

namespace hpfrec
{
    namespace
    {
        cv::CascadeClassifier *GetFaceCascade()
        {
            static std::once_flag loadOnce;
            static cv::CascadeClassifier cascade;
            static bool loaded = false;

            std::call_once(loadOnce, []()
                           {
                const std::array<std::string, 5> candidates = {
                    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
                    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                    cv::samples::findFile("haarcascades/haarcascade_frontalface_default.xml", false),
                    cv::samples::findFile("haarcascade_frontalface_default.xml", false)
                };

                for (const auto &candidate : candidates)
                {
                    if (candidate.empty())
                    {
                        continue;
                    }

                    if (std::filesystem::exists(candidate) && cascade.load(candidate))
                    {
                        loaded = true;
                        return;
                    }
                } });

            return loaded ? &cascade : nullptr;
        }

        cv::Rect MakeSquareFaceRect(const cv::Rect &faceRect, const cv::Size &imageSize)
        {
            const int side = std::max(faceRect.width, faceRect.height);
            const int expandedSide = std::min(std::max(side + side / 5, side), std::min(imageSize.width, imageSize.height));
            const int centerX = faceRect.x + faceRect.width / 2;
            const int centerY = faceRect.y + faceRect.height / 2;

            int x = centerX - expandedSide / 2;
            int y = centerY - expandedSide / 2;

            x = std::max(0, std::min(x, imageSize.width - expandedSide));
            y = std::max(0, std::min(y, imageSize.height - expandedSide));

            return cv::Rect(x, y, expandedSide, expandedSide);
        }

        cv::Rect DetectPrimaryFaceRect(const cv::Mat &input)
        {
            if (input.empty())
            {
                return cv::Rect();
            }

            cv::CascadeClassifier *cascade = GetFaceCascade();
            if (cascade == nullptr || cascade->empty())
            {
                return cv::Rect();
            }

            cv::Mat gray;
            if (input.channels() == 1)
            {
                gray = input;
            }
            else
            {
                cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
            }

            cv::equalizeHist(gray, gray);

            std::vector<cv::Rect> faces;
            cascade->detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(30, 30));
            if (faces.empty())
            {
                return cv::Rect();
            }

            return *std::max_element(faces.begin(), faces.end(), [](const cv::Rect &left, const cv::Rect &right)
                                     { return left.area() < right.area(); });
        }

        cv::Mat CropFaceSquare(const cv::Mat &image)
        {
            const cv::Rect faceRect = DetectPrimaryFaceRect(image);
            if (faceRect.width <= 0 || faceRect.height <= 0)
            {
                return image.clone();
            }

            const cv::Rect squareRect = MakeSquareFaceRect(faceRect, image.size());
            return image(squareRect).clone();
        }

        cv::Mat DrawFacePreview(const cv::Mat &image)
        {
            cv::Mat preview = image.clone();
            const cv::Rect faceRect = DetectPrimaryFaceRect(preview);
            if (faceRect.width > 0 && faceRect.height > 0)
            {
                const cv::Rect squareRect = MakeSquareFaceRect(faceRect, preview.size());
                cv::rectangle(preview, squareRect, cv::Scalar(0, 255, 0), 3);
            }

            return preview;
        }
    }

    cv::Mat LoadPreparedImage(const std::string &path)
    {
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty())
        {
            return cv::Mat();
        }

        image = CropFaceSquare(image);
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, cv::Size(kTrainingWidth, kTrainingHeight));
        cv::equalizeHist(gray, gray);
        gray.convertTo(gray, CV_32F, 1.0 / 255.0);
        return gray;
    }

    cv::Mat LoadPreviewImage(const std::string &path)
    {
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty())
        {
            return cv::Mat();
        }

        return DrawFacePreview(image);
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
