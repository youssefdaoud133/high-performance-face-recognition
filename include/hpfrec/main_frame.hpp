#pragma once

#include "hpfrec/face_database.hpp"

#include <wx/wx.h>

#include <string>
#include <utility>
#include <vector>

namespace hpfrec
{
    class MainFrame : public wxFrame
    {
    public:
        MainFrame();

    private:
        struct EvaluationSample
        {
            bool trueKnown = false;
            bool predictedKnown = false;
            float score = 0.0f;
        };

        void AppendLog(const wxString &message);
        void UpdateStatus();
        void UpdateEvaluationViews();
        void ShowPreview(const std::string &path);
        void AskForFeedback(const std::string &prediction, float distance);
        void RegisterFeedback(bool userConfirmed, const std::string &prediction, float distance);
        std::vector<std::pair<float, float>> BuildRocPoints() const;
        float ComputeAuc(const std::vector<std::pair<float, float>> &points) const;
        wxString BuildMetricsText() const;
        wxBitmap BuildRocBitmap(int width, int height) const;
        void OnAddTrainingImages(wxCommandEvent &);
        void OnTrainModel(wxCommandEvent &);
        void OnRecognizeImage(wxCommandEvent &);

        FaceDatabase model;
        wxTextCtrl *personName = nullptr;
        wxButton *addTrainingButton = nullptr;
        wxButton *trainButton = nullptr;
        wxButton *recognizeButton = nullptr;
        wxStaticBitmap *preview = nullptr;
        wxStaticBitmap *rocPlot = nullptr;
        wxStaticText *statusText = nullptr;
        wxStaticText *resultText = nullptr;
        wxTextCtrl *metricsBox = nullptr;
        wxTextCtrl *logBox = nullptr;

        std::vector<EvaluationSample> evaluationSamples;
        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;
    };
}
