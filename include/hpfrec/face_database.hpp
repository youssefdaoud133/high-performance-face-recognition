#pragma once

#include <opencv2/opencv.hpp>

#include <map>
#include <limits>
#include <string>
#include <vector>

namespace hpfrec
{
    class FaceDatabase
    {
    public:
        bool AddTrainingImages(const std::string &person, const std::vector<std::string> &paths, std::string &error);
        bool Train(std::string &error);

        bool IsTrained() const;
        int PersonCount() const;
        int SampleCount() const;
        size_t ComponentCount() const;
        std::string Recognize(const std::string &path, float *distanceOut = nullptr) const;
        std::string Summary() const;

    private:
        struct TrainingSample
        {
            std::string person;
            std::string path;
            cv::Mat imageVector;
            cv::Mat featureVector;
        };

        static int SelectComponentCount(const cv::Mat &eigenValuesRow);
        cv::Mat ProjectVector(const cv::Mat &vector) const;
        void BuildPersonCentroids();
        void BuildRejectionThreshold();

        std::vector<TrainingSample> trainingSamples;
        cv::Mat meanVector;
        std::vector<cv::Mat> principalEigenVectors;
        std::vector<float> principalEigenValues;
        std::map<std::string, cv::Mat> personCentroids;
        float rejectionThreshold = std::numeric_limits<float>::max();
        bool trained = false;
    };
}
