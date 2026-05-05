#include "hpfrec/app.hpp"

#include "hpfrec/main_frame.hpp"

bool MyApp::OnInit()
{
    auto *frame = new hpfrec::MainFrame();
    frame->Show();
    return true;
}

wxIMPLEMENT_APP(MyApp);
