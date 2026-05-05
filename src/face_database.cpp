#include "hpfrec/face_database.hpp"

#include "hpfrec/image_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>

namespace hpfrec
{
    namespace
    {
        constexpr float kVarianceKeepRatio = 0.90f;

        float L2Distance(const cv::Mat &a, const cv::Mat &b)
        {
            cv::Mat diff = a - b;
            return static_cast<float>(cv::norm(diff, cv::NORM_L2));
        }
    }

    bool FaceDatabase::AddTrainingImages(const std::string &person, const std::vector<std::string> &paths, std::string &error)
    {
        if (person.empty())
        {
            error = "Person name is required.";
            return false;
        }

        if (paths.empty())
        {
            error = "Select at least one image.";
            return false;
        }

        std::size_t added = 0;
        for (const auto &path : paths)
        {
            cv::Mat prepared = LoadPreparedImage(path);
            if (prepared.empty())
            {
                continue;
            }

            TrainingSample sample;
            sample.person = person;
            sample.path = path;
            sample.imageVector = FlattenImage(prepared);
            trainingSamples.push_back(std::move(sample));
            ++added;
        }

        if (added == 0)
        {
            error = "None of the selected files could be loaded as images.";
            return false;
        }

        trained = false;
        return true;
    }

    bool FaceDatabase::Train(std::string &error)
    {
        if (trainingSamples.size() < 2)
        {
            error = "Add at least two face images before training.";
            return false;
        }

        const int pixelsPerImage = static_cast<int>(trainingSamples.front().imageVector.total());
        for (const auto &sample : trainingSamples)
        {
            if (static_cast<int>(sample.imageVector.total()) != pixelsPerImage)
            {
                error = "All training images must have the same dimensions.";
                return false;
            }
        }

        const int imageCount = static_cast<int>(trainingSamples.size());
        cv::Mat dataMatrix(pixelsPerImage, imageCount, CV_32F);
        for (int columnIndex = 0; columnIndex < imageCount; ++columnIndex)
        {
            trainingSamples[columnIndex].imageVector.copyTo(dataMatrix.col(columnIndex));
        }

        cv::reduce(dataMatrix, meanVector, 1, cv::REDUCE_AVG);

        cv::Mat centered = dataMatrix.clone();
        for (int columnIndex = 0; columnIndex < imageCount; ++columnIndex)
        {
            centered.col(columnIndex) -= meanVector;
        }

        cv::Mat covarianceMatrix = (centered.t() * centered) / static_cast<float>(std::max(1, imageCount - 1));
        cv::Mat eigenValuesRow;
        cv::Mat eigenVectorsRow;
        if (!cv::eigen(covarianceMatrix, eigenValuesRow, eigenVectorsRow))
        {
            error = "Unable to solve eigenvectors for the training data.";
            return false;
        }

        cv::Mat eigenValues;
        eigenValuesRow.convertTo(eigenValues, CV_32F);

        const int componentCount = SelectComponentCount(eigenValues);
        if (componentCount <= 0)
        {
            error = "No principal components were retained.";
            return false;
        }

        principalEigenVectors.clear();
        principalEigenValues.clear();
        principalEigenVectors.reserve(componentCount);
        principalEigenValues.reserve(componentCount);

        for (int index = 0; index < componentCount; ++index)
        {
            float eigenValue = 0.0f;
            if (eigenValues.rows == 1)
            {
                eigenValue = eigenValues.at<float>(0, index);
            }
            else
            {
                eigenValue = eigenValues.at<float>(index, 0);
            }

            if (eigenValue <= std::numeric_limits<float>::epsilon())
            {
                continue;
            }

            cv::Mat sampleSpaceVector64 = eigenVectorsRow.row(index).t();
            cv::Mat sampleSpaceVector;
            sampleSpaceVector64.convertTo(sampleSpaceVector, CV_32F);

            cv::Mat faceSpaceVector = centered * sampleSpaceVector;
            faceSpaceVector /= static_cast<float>(std::sqrt(static_cast<double>(eigenValue)));
            cv::normalize(faceSpaceVector, faceSpaceVector);

            principalEigenVectors.push_back(faceSpaceVector);
            principalEigenValues.push_back(eigenValue);
        }

        if (principalEigenVectors.empty())
        {
            error = "No usable eigenvectors were produced.";
            return false;
        }

        for (auto &sample : trainingSamples)
        {
            sample.featureVector = ProjectVector(sample.imageVector);
        }

        BuildPersonCentroids();
        BuildRejectionThreshold();

        trained = true;
        error.clear();
        return true;
    }

    bool FaceDatabase::IsTrained() const
    {
        return trained;
    }

    int FaceDatabase::PersonCount() const
    {
        std::map<std::string, int> counts;
        for (const auto &sample : trainingSamples)
        {
            ++counts[sample.person];
        }
        return static_cast<int>(counts.size());
    }

    int FaceDatabase::SampleCount() const
    {
        return static_cast<int>(trainingSamples.size());
    }

    size_t FaceDatabase::ComponentCount() const
    {
        return principalEigenVectors.size();
    }

    std::string FaceDatabase::Recognize(const std::string &path, float *distanceOut) const
    {
        if (!trained)
        {
            return "Model is not trained yet.";
        }

        cv::Mat image = LoadPreparedImage(path);
        if (image.empty())
        {
            return "Unable to load the selected image.";
        }

        cv::Mat feature = ProjectVector(FlattenImage(image));

        std::string bestPerson = "Unknown";
        float bestDistance = std::numeric_limits<float>::max();
        for (const auto &entry : personCentroids)
        {
            const float distance = L2Distance(feature, entry.second);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestPerson = entry.first;
            }
        }

        if (distanceOut != nullptr)
        {
            *distanceOut = bestDistance;
        }

        if (bestDistance > rejectionThreshold)
        {
            return "Unknown";
        }

        return bestPerson;
    }

    std::string FaceDatabase::Summary() const
    {
        std::ostringstream stream;
        stream << "Training images: " << SampleCount() << "\n";
        stream << "People: " << PersonCount() << "\n";
        stream << "Principal components: " << ComponentCount() << "\n";
        stream << "Trained: " << (trained ? "yes" : "no");
        return stream.str();
    }

    int FaceDatabase::SelectComponentCount(const cv::Mat &eigenValuesRow)
    {
        const int totalComponents = static_cast<int>(eigenValuesRow.total());
        if (totalComponents == 0)
        {
            return 0;
        }

        double totalVariance = 0.0;
        for (int i = 0; i < totalComponents; ++i)
        {
            totalVariance += eigenValuesRow.at<float>(0, i);
        }

        if (totalVariance <= std::numeric_limits<double>::epsilon())
        {
            return 1;
        }

        double cumulativeVariance = 0.0;
        for (int i = 0; i < totalComponents; ++i)
        {
            cumulativeVariance += eigenValuesRow.at<float>(0, i);
            if ((cumulativeVariance / totalVariance) >= kVarianceKeepRatio)
            {
                return i + 1;
            }
        }

        return totalComponents;
    }

    cv::Mat FaceDatabase::ProjectVector(const cv::Mat &vector) const
    {
        cv::Mat centered = vector - meanVector;
        cv::Mat feature(static_cast<int>(principalEigenVectors.size()), 1, CV_32F, cv::Scalar(0));
        for (int i = 0; i < static_cast<int>(principalEigenVectors.size()); ++i)
        {
            feature.at<float>(i, 0) = static_cast<float>(principalEigenVectors[i].dot(centered));
        }
        return feature;
    }

    void FaceDatabase::BuildPersonCentroids()
    {
        std::map<std::string, std::vector<cv::Mat>> grouped;
        for (const auto &sample : trainingSamples)
        {
            grouped[sample.person].push_back(sample.featureVector);
        }

        personCentroids.clear();
        for (const auto &entry : grouped)
        {
            cv::Mat centroid = cv::Mat::zeros(entry.second.front().rows, 1, CV_32F);
            for (const auto &feature : entry.second)
            {
                centroid += feature;
            }
            centroid /= static_cast<float>(entry.second.size());
            personCentroids[entry.first] = centroid;
        }
    }

    void FaceDatabase::BuildRejectionThreshold()
    {
        std::vector<float> intraClassDistances;
        intraClassDistances.reserve(trainingSamples.size());

        for (const auto &sample : trainingSamples)
        {
            const auto centroidIt = personCentroids.find(sample.person);
            if (centroidIt == personCentroids.end())
            {
                continue;
            }

            intraClassDistances.push_back(L2Distance(sample.featureVector, centroidIt->second));
        }

        if (intraClassDistances.empty())
        {
            rejectionThreshold = std::numeric_limits<float>::max();
            return;
        }

        const float averageDistance = std::accumulate(intraClassDistances.begin(), intraClassDistances.end(), 0.0f) /
                                      static_cast<float>(intraClassDistances.size());
        const float maximumDistance = *std::max_element(intraClassDistances.begin(), intraClassDistances.end());

        rejectionThreshold = std::max(maximumDistance * 1.15f, averageDistance * 1.75f);
        rejectionThreshold = std::max(rejectionThreshold, 0.35f);
    }
}
