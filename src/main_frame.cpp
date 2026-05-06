#include "hpfrec/main_frame.hpp"

#include "hpfrec/image_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

namespace hpfrec
{
    MainFrame::MainFrame()
        : wxFrame(nullptr, wxID_ANY, "High Performance Face Recognition", wxDefaultPosition, wxSize(1080, 760))
    {
        auto *root = new wxPanel(this);

        auto *title = new wxStaticText(root, wxID_ANY, "Eigenfaces training and recognition");
        wxFont titleFont = title->GetFont();
        titleFont.SetPointSize(titleFont.GetPointSize() + 4);
        titleFont.SetWeight(wxFONTWEIGHT_BOLD);
        title->SetFont(titleFont);

        personName = new wxTextCtrl(root, wxID_ANY);
        personName->SetHint("Person name for the selected training images");

        addTrainingButton = new wxButton(root, wxID_ANY, "Upload Training Photos");
        trainButton = new wxButton(root, wxID_ANY, "Train Model");
        recognizeButton = new wxButton(root, wxID_ANY, "Upload New Image");

        preview = new wxStaticBitmap(root, wxID_ANY, MatToBitmap(cv::Mat()));
        preview->SetMinSize(wxSize(320, 320));

        rocPlot = new wxStaticBitmap(root, wxID_ANY, wxBitmap(420, 260));
        rocPlot->SetMinSize(wxSize(420, 260));

        statusText = new wxStaticText(root, wxID_ANY, "Add face images to start training.");
        resultText = new wxStaticText(root, wxID_ANY, "Result: waiting for recognition");

        metricsBox = new wxTextCtrl(root, wxID_ANY, "", wxDefaultPosition, wxSize(420, 185),
                                    wxTE_MULTILINE | wxTE_READONLY | wxTE_BESTWRAP);

        logBox = new wxTextCtrl(root, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                wxTE_MULTILINE | wxTE_READONLY | wxTE_BESTWRAP);

        auto *controls = new wxBoxSizer(wxVERTICAL);
        controls->Add(title, 0, wxBOTTOM, 10);
        controls->Add(new wxStaticText(root, wxID_ANY, "Person name:"), 0, wxBOTTOM, 4);
        controls->Add(personName, 0, wxEXPAND | wxBOTTOM, 10);
        controls->Add(addTrainingButton, 0, wxEXPAND | wxBOTTOM, 6);
        controls->Add(trainButton, 0, wxEXPAND | wxBOTTOM, 6);
        controls->Add(recognizeButton, 0, wxEXPAND | wxBOTTOM, 14);
        controls->Add(statusText, 0, wxBOTTOM, 12);
        controls->Add(resultText, 0, wxBOTTOM, 12);
        controls->Add(new wxStaticText(root, wxID_ANY, "Log:"), 0, wxBOTTOM, 4);
        controls->Add(logBox, 1, wxEXPAND);

        auto *analysisSizer = new wxBoxSizer(wxVERTICAL);
        analysisSizer->Add(new wxStaticText(root, wxID_ANY, "Selected image preview"), 0, wxBOTTOM, 4);
        analysisSizer->Add(preview, 0, wxEXPAND | wxBOTTOM, 12);
        analysisSizer->Add(new wxStaticText(root, wxID_ANY, "ROC curve (updated from user feedback)"), 0, wxBOTTOM, 4);
        analysisSizer->Add(rocPlot, 0, wxEXPAND | wxBOTTOM, 8);
        analysisSizer->Add(new wxStaticText(root, wxID_ANY, "Performance matrix and metrics"), 0, wxBOTTOM, 4);
        analysisSizer->Add(metricsBox, 0, wxEXPAND);

        auto *content = new wxBoxSizer(wxHORIZONTAL);
        content->Add(controls, 1, wxEXPAND | wxALL, 16);
        content->Add(analysisSizer, 0, wxEXPAND | wxTOP | wxRIGHT | wxBOTTOM, 16);

        root->SetSizer(content);

        addTrainingButton->Bind(wxEVT_BUTTON, &MainFrame::OnAddTrainingImages, this);
        trainButton->Bind(wxEVT_BUTTON, &MainFrame::OnTrainModel, this);
        recognizeButton->Bind(wxEVT_BUTTON, &MainFrame::OnRecognizeImage, this);

        UpdateStatus();
        UpdateEvaluationViews();
        AppendLog("Recommended setup: 5 people with 10 labeled face photos each.");
        AppendLog("Upload training images first, then train the model, then upload a new image to recognize.");
        AppendLog("After every recognition, confirm whether prediction is correct to update ROC and performance metrics.");
    }

    void MainFrame::AppendLog(const wxString &message)
    {
        logBox->AppendText(message + "\n");
    }

    void MainFrame::UpdateStatus()
    {
        statusText->SetLabel(wxString::Format("Training images: %d | People: %d | Trained: %s",
                                              model.SampleCount(), model.PersonCount(),
                                              model.IsTrained() ? wxString("yes") : wxString("no")));
        Layout();
    }

    void MainFrame::UpdateEvaluationViews()
    {
        metricsBox->SetValue(BuildMetricsText());
        rocPlot->SetBitmap(BuildRocBitmap(420, 260));
        Layout();
    }

    void MainFrame::ShowPreview(const std::string &path)
    {
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty())
        {
            preview->SetBitmap(MatToBitmap(cv::Mat()));
            return;
        }

        cv::resize(image, image, cv::Size(320, 320));
        preview->SetBitmap(MatToBitmap(image));
        Layout();
    }

    void MainFrame::OnAddTrainingImages(wxCommandEvent &)
    {
        const std::string person = personName->GetValue().ToStdString();
        if (person.empty())
        {
            wxMessageBox("Enter a person name before uploading images.", "Missing person name", wxOK | wxICON_WARNING, this);
            return;
        }

        wxFileDialog dialog(this, "Select training face photos", wxEmptyString, wxEmptyString,
                            "Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp",
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_MULTIPLE);

        if (dialog.ShowModal() != wxID_OK)
        {
            return;
        }

        wxArrayString selectedPaths;
        dialog.GetPaths(selectedPaths);

        std::vector<std::string> paths;
        paths.reserve(selectedPaths.size());
        for (const auto &selectedPath : selectedPaths)
        {
            paths.push_back(selectedPath.ToStdString());
        }

        std::string error;
        if (!model.AddTrainingImages(person, paths, error))
        {
            wxMessageBox(wxString::FromUTF8(error), "Training upload failed", wxOK | wxICON_ERROR, this);
            return;
        }

        AppendLog(wxString::Format("Added %lu training image(s) for %s.",
                                   static_cast<unsigned long>(paths.size()), person.c_str()));
        UpdateStatus();
    }

    void MainFrame::OnTrainModel(wxCommandEvent &)
    {
        std::string error;
        if (!model.Train(error))
        {
            wxMessageBox(wxString::FromUTF8(error), "Training failed", wxOK | wxICON_ERROR, this);
            return;
        }

        AppendLog("Model trained successfully.");
        AppendLog(wxString::Format("Retained principal components: %lu",
                                   static_cast<unsigned long>(model.ComponentCount())));
        AppendLog(wxString::FromUTF8(model.Summary()));
        UpdateStatus();
    }

    void MainFrame::OnRecognizeImage(wxCommandEvent &)
    {
        if (!model.IsTrained())
        {
            wxMessageBox("Train the model before uploading a new image for recognition.", "Model not trained", wxOK | wxICON_WARNING, this);
            return;
        }

        wxFileDialog dialog(this, "Select a face image to recognize", wxEmptyString, wxEmptyString,
                            "Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp",
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST);

        if (dialog.ShowModal() != wxID_OK)
        {
            return;
        }

        const std::string path = dialog.GetPath().ToStdString();
        ShowPreview(path);

        float distance = 0.0f;
        const std::string result = model.Recognize(path, &distance);
        if (result == "Unknown")
        {
            AppendLog(wxString::Format("Recognition result: Unknown (distance %.4f)", distance));
            resultText->SetLabel("Result: Unknown person");
        }
        else if (result.rfind("Model is not trained yet.", 0) == 0 || result.rfind("Unable to load", 0) == 0)
        {
            wxMessageBox(wxString::FromUTF8(result), "Recognition failed", wxOK | wxICON_ERROR, this);
            return;
        }
        else
        {
            AppendLog(wxString::Format("Recognition result: %s (distance %.4f)", result.c_str(), distance));
            resultText->SetLabel(wxString::Format("Result: %s", result.c_str()));
        }

        AskForFeedback(result, distance);
    }

    void MainFrame::AskForFeedback(const std::string &prediction, float distance)
    {
        wxString message;
        if (prediction == "Unknown")
        {
            message = wxString::Format("Prediction is Unknown (distance %.4f). Is this correct?", distance);
        }
        else
        {
            message = wxString::Format("Prediction is %s (distance %.4f). Is this correct?",
                                       prediction.c_str(), distance);
        }

        wxMessageDialog dialog(this, message, "Feedback", wxYES_NO | wxCANCEL | wxICON_QUESTION);
        const int decision = dialog.ShowModal();
        if (decision == wxID_CANCEL)
        {
            AppendLog("Feedback skipped. Metrics and ROC unchanged.");
            return;
        }

        RegisterFeedback(decision == wxID_YES, prediction, distance);
    }

    void MainFrame::RegisterFeedback(bool userConfirmed, const std::string &prediction, float distance)
    {
        const bool predictedKnown = prediction != "Unknown";
        const bool trueKnown = userConfirmed ? predictedKnown : !predictedKnown;

        evaluationSamples.push_back(EvaluationSample{trueKnown, predictedKnown, -distance});

        if (predictedKnown && trueKnown)
        {
            ++truePositives;
        }
        else if (predictedKnown && !trueKnown)
        {
            ++falsePositives;
        }
        else if (!predictedKnown && trueKnown)
        {
            ++falseNegatives;
        }
        else
        {
            ++trueNegatives;
        }

        AppendLog(wxString::Format("Feedback saved: %s", userConfirmed ? "prediction confirmed" : "prediction corrected"));
        UpdateEvaluationViews();
    }

    std::vector<std::pair<float, float>> MainFrame::BuildRocPoints() const
    {
        std::vector<std::pair<float, float>> points;
        if (evaluationSamples.empty())
        {
            return points;
        }

        std::vector<float> thresholds;
        thresholds.reserve(evaluationSamples.size() + 2);

        float minScore = evaluationSamples.front().score;
        float maxScore = evaluationSamples.front().score;
        for (const auto &sample : evaluationSamples)
        {
            thresholds.push_back(sample.score);
            minScore = std::min(minScore, sample.score);
            maxScore = std::max(maxScore, sample.score);
        }

        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

        thresholds.insert(thresholds.begin(), maxScore + 1.0f);
        thresholds.push_back(minScore - 1.0f);

        for (const float threshold : thresholds)
        {
            int tp = 0;
            int fp = 0;
            int tn = 0;
            int fn = 0;

            for (const auto &sample : evaluationSamples)
            {
                const bool predictedPositive = sample.score >= threshold;
                if (predictedPositive && sample.trueKnown)
                {
                    ++tp;
                }
                else if (predictedPositive && !sample.trueKnown)
                {
                    ++fp;
                }
                else if (!predictedPositive && sample.trueKnown)
                {
                    ++fn;
                }
                else
                {
                    ++tn;
                }
            }

            const float tpr = (tp + fn) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
            const float fpr = (fp + tn) > 0 ? static_cast<float>(fp) / static_cast<float>(fp + tn) : 0.0f;
            points.emplace_back(fpr, tpr);
        }

        std::sort(points.begin(), points.end(), [](const auto &left, const auto &right)
                  {
            if (left.first == right.first)
            {
                return left.second < right.second;
            }
            return left.first < right.first; });

        return points;
    }

    float MainFrame::ComputeAuc(const std::vector<std::pair<float, float>> &points) const
    {
        if (points.size() < 2)
        {
            return 0.0f;
        }

        float area = 0.0f;
        for (std::size_t index = 1; index < points.size(); ++index)
        {
            const float x0 = points[index - 1].first;
            const float y0 = points[index - 1].second;
            const float x1 = points[index].first;
            const float y1 = points[index].second;
            area += (x1 - x0) * (y0 + y1) * 0.5f;
        }

        return std::clamp(area, 0.0f, 1.0f);
    }

    wxString MainFrame::BuildMetricsText() const
    {
        const int total = truePositives + falsePositives + trueNegatives + falseNegatives;
        const auto ratio = [](int numerator, int denominator) -> double
        {
            if (denominator == 0)
            {
                return 0.0;
            }
            return static_cast<double>(numerator) / static_cast<double>(denominator);
        };

        const double accuracy = ratio(truePositives + trueNegatives, total);
        const double precision = ratio(truePositives, truePositives + falsePositives);
        const double recall = ratio(truePositives, truePositives + falseNegatives);
        const double specificity = ratio(trueNegatives, trueNegatives + falsePositives);
        const double falsePositiveRate = ratio(falsePositives, falsePositives + trueNegatives);
        const double f1 = (precision + recall) > 0.0 ? (2.0 * precision * recall) / (precision + recall) : 0.0;
        const double auc = static_cast<double>(ComputeAuc(BuildRocPoints()));

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(4);
        stream << "Confusion Matrix (Known vs Unknown)\n";
        stream << "TP=" << truePositives << "  FP=" << falsePositives << "\n";
        stream << "FN=" << falseNegatives << "  TN=" << trueNegatives << "\n\n";
        stream << "Samples with feedback: " << total << "\n";
        stream << "Accuracy: " << accuracy << "\n";
        stream << "Precision: " << precision << "\n";
        stream << "Recall (TPR): " << recall << "\n";
        stream << "Specificity (TNR): " << specificity << "\n";
        stream << "False Positive Rate: " << falsePositiveRate << "\n";
        stream << "F1 score: " << f1 << "\n";
        stream << "ROC AUC: " << auc << "\n";

        return wxString::FromUTF8(stream.str());
    }

    wxBitmap MainFrame::BuildRocBitmap(int width, int height) const
    {
        wxBitmap bitmap(width, height);
        wxMemoryDC dc(bitmap);
        dc.SetBackground(*wxWHITE_BRUSH);
        dc.Clear();

        const int leftMargin = 48;
        const int rightMargin = 16;
        const int topMargin = 18;
        const int bottomMargin = 36;
        const int plotWidth = std::max(1, width - leftMargin - rightMargin);
        const int plotHeight = std::max(1, height - topMargin - bottomMargin);

        dc.SetPen(wxPen(wxColour(230, 230, 230), 1));
        for (int tick = 0; tick <= 5; ++tick)
        {
            const int x = leftMargin + (plotWidth * tick) / 5;
            const int y = topMargin + (plotHeight * tick) / 5;
            dc.DrawLine(x, topMargin, x, topMargin + plotHeight);
            dc.DrawLine(leftMargin, y, leftMargin + plotWidth, y);
        }

        dc.SetPen(wxPen(wxColour(40, 40, 40), 1));
        dc.DrawLine(leftMargin, topMargin, leftMargin, topMargin + plotHeight);
        dc.DrawLine(leftMargin, topMargin + plotHeight, leftMargin + plotWidth, topMargin + plotHeight);

        dc.SetTextForeground(wxColour(45, 45, 45));
        dc.DrawText("0.0", leftMargin - 12, topMargin + plotHeight + 4);
        dc.DrawText("1.0", leftMargin + plotWidth - 10, topMargin + plotHeight + 4);
        dc.DrawText("1.0", leftMargin - 30, topMargin - 6);
        dc.DrawText("FPR", leftMargin + (plotWidth / 2) - 10, topMargin + plotHeight + 18);
        dc.DrawText("TPR", 6, topMargin + (plotHeight / 2) - 8);

        dc.SetPen(wxPen(wxColour(165, 165, 165), 1, wxPENSTYLE_SHORT_DASH));
        dc.DrawLine(leftMargin, topMargin + plotHeight, leftMargin + plotWidth, topMargin);

        const auto points = BuildRocPoints();
        if (points.empty())
        {
            dc.SetTextForeground(wxColour(95, 95, 95));
            dc.DrawText("No feedback samples yet", leftMargin + 72, topMargin + (plotHeight / 2));
            dc.SelectObject(wxNullBitmap);
            return bitmap;
        }

        std::vector<wxPoint> rocPolyline;
        rocPolyline.reserve(points.size());
        for (const auto &point : points)
        {
            const int x = leftMargin + static_cast<int>(std::round(point.first * static_cast<float>(plotWidth)));
            const int y = topMargin + static_cast<int>(std::round((1.0f - point.second) * static_cast<float>(plotHeight)));
            rocPolyline.emplace_back(x, y);
        }

        dc.SetPen(wxPen(wxColour(20, 116, 217), 2));
        dc.DrawLines(static_cast<int>(rocPolyline.size()), rocPolyline.data());

        std::ostringstream aucLabel;
        aucLabel << std::fixed << std::setprecision(4) << "AUC=" << ComputeAuc(points);
        dc.SetTextForeground(wxColour(20, 116, 217));
        dc.DrawText(wxString::FromUTF8(aucLabel.str()), leftMargin + plotWidth - 80, topMargin + 2);

        dc.SelectObject(wxNullBitmap);
        return bitmap;
    }
}
