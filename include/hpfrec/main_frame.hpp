#pragma once

#include "hpfrec/face_database.hpp"

#include <wx/wx.h>

namespace hpfrec
{
    class MainFrame : public wxFrame
    {
    public:
        MainFrame();

    private:
        void AppendLog(const wxString &message);
        void UpdateStatus();
        void ShowPreview(const std::string &path);
        void OnAddTrainingImages(wxCommandEvent &);
        void OnTrainModel(wxCommandEvent &);
        void OnRecognizeImage(wxCommandEvent &);

        FaceDatabase model;
        wxTextCtrl *personName = nullptr;
        wxButton *addTrainingButton = nullptr;
        wxButton *trainButton = nullptr;
        wxButton *recognizeButton = nullptr;
        wxStaticBitmap *preview = nullptr;
        wxStaticText *statusText = nullptr;
        wxTextCtrl *logBox = nullptr;
    };
}
