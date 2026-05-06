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
        constexpr int kPowerIterationMaxSteps = 128;
        constexpr float kPowerIterationTolerance = 1e-4f;
        constexpr float kLinearAlgebraEpsilon = 1e-8f;

        float L2Distance(const cv::Mat &a, const cv::Mat &b)
        {
            cv::Mat diff = a - b;
            return static_cast<float>(cv::norm(diff, cv::NORM_L2));
        }

        cv::Mat BuildMeanVector(const cv::Mat &dataMatrix)
        {
            cv::Mat mean = cv::Mat::zeros(dataMatrix.rows, 1, CV_32F);
            const int imageCount = dataMatrix.cols;

            for (int row = 0; row < dataMatrix.rows; ++row)
            {
                float sum = 0.0f;
                for (int column = 0; column < imageCount; ++column)
                {
                    sum += dataMatrix.at<float>(row, column);
                }
                mean.at<float>(row, 0) = sum / static_cast<float>(imageCount);
            }

            return mean;
        }

        cv::Mat BuildCenteredMatrix(const cv::Mat &dataMatrix, const cv::Mat &meanVector)
        {
            cv::Mat centered = cv::Mat::zeros(dataMatrix.rows, dataMatrix.cols, CV_32F);
            for (int row = 0; row < dataMatrix.rows; ++row)
            {
                const float meanValue = meanVector.at<float>(row, 0);
                for (int column = 0; column < dataMatrix.cols; ++column)
                {
                    centered.at<float>(row, column) = dataMatrix.at<float>(row, column) - meanValue;
                }
            }
            return centered;
        }

        cv::Mat BuildCovarianceMatrix(const cv::Mat &centered)
        {
            const int imageCount = centered.cols;
            cv::Mat covariance = cv::Mat::zeros(imageCount, imageCount, CV_32F);
            const float scale = 1.0f / static_cast<float>(std::max(1, imageCount - 1));

            for (int i = 0; i < imageCount; ++i)
            {
                for (int j = i; j < imageCount; ++j)
                {
                    float sum = 0.0f;
                    for (int row = 0; row < centered.rows; ++row)
                    {
                        sum += centered.at<float>(row, i) * centered.at<float>(row, j);
                    }

                    const float value = sum * scale;
                    covariance.at<float>(i, j) = value;
                    covariance.at<float>(j, i) = value;
                }
            }

            return covariance;
        }

        std::vector<float> MultiplyMatrixVector(const cv::Mat &matrix, const std::vector<float> &vector)
        {
            std::vector<float> result(matrix.rows, 0.0f);
            for (int row = 0; row < matrix.rows; ++row)
            {
                float sum = 0.0f;
                for (int column = 0; column < matrix.cols; ++column)
                {
                    sum += matrix.at<float>(row, column) * vector[column];
                }
                result[row] = sum;
            }
            return result;
        }

        cv::Mat MultiplyMatrixVectorAsColumn(const cv::Mat &matrix, const std::vector<float> &vector)
        {
            cv::Mat result = cv::Mat::zeros(matrix.rows, 1, CV_32F);
            for (int row = 0; row < matrix.rows; ++row)
            {
                float sum = 0.0f;
                for (int column = 0; column < matrix.cols; ++column)
                {
                    sum += matrix.at<float>(row, column) * vector[column];
                }
                result.at<float>(row, 0) = sum;
            }
            return result;
        }

        float VectorNorm(const std::vector<float> &vector)
        {
            float sum = 0.0f;
            for (float value : vector)
            {
                sum += value * value;
            }
            return std::sqrt(sum);
        }

        float VectorDifferenceNorm(const std::vector<float> &left, const std::vector<float> &right)
        {
            float sum = 0.0f;
            for (std::size_t index = 0; index < left.size(); ++index)
            {
                const float delta = left[index] - right[index];
                sum += delta * delta;
            }
            return std::sqrt(sum);
        }

        void NormalizeVector(std::vector<float> &vector)
        {
            const float norm = VectorNorm(vector);
            if (norm <= kLinearAlgebraEpsilon)
            {
                return;
            }

            for (float &value : vector)
            {
                value /= norm;
            }
        }

        float RayleighQuotient(const cv::Mat &matrix, const std::vector<float> &vector)
        {
            const std::vector<float> multiplied = MultiplyMatrixVector(matrix, vector);
            float result = 0.0f;
            for (std::size_t index = 0; index < vector.size(); ++index)
            {
                result += vector[index] * multiplied[index];
            }
            return result;
        }

        void DeflateMatrix(cv::Mat &matrix, float eigenValue, const std::vector<float> &eigenVector)
        {
            for (int row = 0; row < matrix.rows; ++row)
            {
                for (int column = 0; column < matrix.cols; ++column)
                {
                    matrix.at<float>(row, column) -= eigenValue * eigenVector[row] * eigenVector[column];
                }
            }
        }

        std::vector<float> PowerIteration(const cv::Mat &matrix, std::vector<float> seed)
        {
            NormalizeVector(seed);
            if (VectorNorm(seed) <= kLinearAlgebraEpsilon)
            {
                seed.assign(matrix.cols, 0.0f);
                if (!seed.empty())
                {
                    seed[0] = 1.0f;
                }
            }

            std::vector<float> vector = seed;
            for (int iteration = 0; iteration < kPowerIterationMaxSteps; ++iteration)
            {
                std::vector<float> next = MultiplyMatrixVector(matrix, vector);
                const float nextNorm = VectorNorm(next);
                if (nextNorm <= kLinearAlgebraEpsilon)
                {
                    break;
                }

                for (float &value : next)
                {
                    value /= nextNorm;
                }

                if (VectorDifferenceNorm(next, vector) <= kPowerIterationTolerance)
                {
                    vector = next;
                    break;
                }

                vector = next;
            }

            NormalizeVector(vector);
            return vector;
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

        meanVector = BuildMeanVector(dataMatrix);
        cv::Mat centered = BuildCenteredMatrix(dataMatrix, meanVector);
        cv::Mat covarianceMatrix = BuildCovarianceMatrix(centered);

        std::vector<float> eigenValues;
        std::vector<std::vector<float>> eigenVectors;
        eigenValues.reserve(imageCount);
        eigenVectors.reserve(imageCount);

        cv::Mat workingCovariance = covarianceMatrix.clone();
        for (int componentIndex = 0; componentIndex < imageCount; ++componentIndex)
        {
            std::vector<float> seed(imageCount, 0.0f);
            seed[componentIndex % imageCount] = 1.0f;

            std::vector<float> eigenVector = PowerIteration(workingCovariance, seed);
            if (VectorNorm(eigenVector) <= kLinearAlgebraEpsilon)
            {
                continue;
            }

            float eigenValue = RayleighQuotient(covarianceMatrix, eigenVector);
            if (eigenValue <= kLinearAlgebraEpsilon)
            {
                continue;
            }

            eigenValues.push_back(eigenValue);
            eigenVectors.push_back(std::move(eigenVector));
            DeflateMatrix(workingCovariance, eigenValue, eigenVectors.back());
        }

        if (eigenValues.empty())
        {
            error = "Unable to derive principal components from the training data.";
            return false;
        }

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
            if (index >= static_cast<int>(eigenValues.size()))
            {
                break;
            }

            const float eigenValue = eigenValues[index];
            if (eigenValue <= std::numeric_limits<float>::epsilon())
            {
                continue;
            }

            cv::Mat faceSpaceVector = MultiplyMatrixVectorAsColumn(centered, eigenVectors[index]);
            const float scale = static_cast<float>(std::sqrt(static_cast<double>(eigenValue)));
            if (scale > std::numeric_limits<float>::epsilon())
            {
                faceSpaceVector /= scale;
            }

            const float faceNorm = static_cast<float>(cv::norm(faceSpaceVector, cv::NORM_L2));
            if (faceNorm > std::numeric_limits<float>::epsilon())
            {
                faceSpaceVector /= faceNorm;
            }

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

    int FaceDatabase::SelectComponentCount(const std::vector<float> &eigenValues)
    {
        const int totalComponents = static_cast<int>(eigenValues.size());
        if (totalComponents == 0)
        {
            return 0;
        }

        double totalVariance = 0.0;
        for (int i = 0; i < totalComponents; ++i)
        {
            totalVariance += eigenValues[i];
        }

        if (totalVariance <= std::numeric_limits<double>::epsilon())
        {
            return 1;
        }

        double cumulativeVariance = 0.0;
        for (int i = 0; i < totalComponents; ++i)
        {
            cumulativeVariance += eigenValues[i];
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
