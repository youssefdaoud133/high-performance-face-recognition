#include "hpfrec/main_frame.hpp"

#include "hpfrec/image_utils.hpp"

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

        statusText = new wxStaticText(root, wxID_ANY, "Add face images to start training.");

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
        controls->Add(new wxStaticText(root, wxID_ANY, "Log:"), 0, wxBOTTOM, 4);
        controls->Add(logBox, 1, wxEXPAND);

        auto *previewSizer = new wxBoxSizer(wxVERTICAL);
        previewSizer->Add(new wxStaticText(root, wxID_ANY, "Selected image preview"), 0, wxBOTTOM, 4);
        previewSizer->Add(preview, 0, wxEXPAND | wxBOTTOM, 8);

        auto *content = new wxBoxSizer(wxHORIZONTAL);
        content->Add(controls, 1, wxEXPAND | wxALL, 16);
        content->Add(previewSizer, 0, wxEXPAND | wxTOP | wxRIGHT | wxBOTTOM, 16);

        root->SetSizer(content);

        addTrainingButton->Bind(wxEVT_BUTTON, &MainFrame::OnAddTrainingImages, this);
        trainButton->Bind(wxEVT_BUTTON, &MainFrame::OnTrainModel, this);
        recognizeButton->Bind(wxEVT_BUTTON, &MainFrame::OnRecognizeImage, this);

        UpdateStatus();
        AppendLog("Recommended setup: 5 people with 10 labeled face photos each.");
        AppendLog("Upload training images first, then train the model, then upload a new image to recognize.");
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
            statusText->SetLabel("Result: Unknown person");
        }
        else if (result.rfind("Model is not trained yet.", 0) == 0 || result.rfind("Unable to load", 0) == 0)
        {
            wxMessageBox(wxString::FromUTF8(result), "Recognition failed", wxOK | wxICON_ERROR, this);
            return;
        }
        else
        {
            AppendLog(wxString::Format("Recognition result: %s (distance %.4f)", result.c_str(), distance));
            statusText->SetLabel(wxString::Format("Result: %s", result.c_str()));
        }
    }
}
