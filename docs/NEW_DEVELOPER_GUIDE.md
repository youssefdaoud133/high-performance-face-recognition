# New Developer Guide

This project is a C++ face-recognition desktop app built with wxWidgets for the UI and OpenCV for image processing and PCA/eigenfaces.

## What The App Does

The app lets a user:

1. Enter a person name.
2. Upload one or more training face images for that person.
3. Train a recognition model.
4. Upload a new face image and get the closest match or `Unknown`.

The current implementation is file-based. It does not use a live camera feed.

## High-Level Flow

The flow is simple:

1. The app starts in `src/app.cpp`.
2. `MyApp::OnInit()` creates the main window from `src/main_frame.cpp`.
3. The user interacts with the UI in `MainFrame`.
4. `MainFrame` calls `FaceDatabase` in `src/face_database.cpp`.
5. `FaceDatabase` loads and prepares images using helpers from `src/image_utils.cpp`.
6. The trained model is used to recognize new images.

## Source Structure

The code is split into small modules so each concern is easy to find:

- `include/hpfrec/app.hpp` and `src/app.cpp`
  - Application entry point
  - Starts wxWidgets and opens the main window

- `include/hpfrec/main_frame.hpp` and `src/main_frame.cpp`
  - Main UI window
  - Buttons, text fields, logs, preview panel
  - Handles user actions: upload training images, train, recognize

- `include/hpfrec/face_database.hpp` and `src/face_database.cpp`
  - Stores training samples in memory
  - Prepares data for PCA/eigenfaces
  - Computes person centroids
  - Applies rejection logic for unknown faces

- `include/hpfrec/image_utils.hpp` and `src/image_utils.cpp`
  - Loads grayscale images
  - Resizes and normalizes them
  - Flattens images into vectors for training
  - Converts OpenCV images into wxWidgets bitmaps for preview

## Runtime Flow In Detail

### 1. App Startup

`src/app.cpp` contains the wxWidgets application object.

`MyApp::OnInit()` creates `hpfrec::MainFrame` and shows it. That is the whole startup path.

### 2. UI Initialization

`src/main_frame.cpp` builds the entire window:

- person name field
- upload training photos button
- train model button
- upload new image button
- image preview panel
- status text
- log output

The window also shows startup guidance in the log so the next step is obvious to the user.

### 3. Adding Training Images

When the user clicks **Upload Training Photos**:

1. The app checks that a person name is entered.
2. A file picker opens.
3. The selected paths are passed into `FaceDatabase::AddTrainingImages()`.
4. Each image is loaded, resized, histogram-equalized, normalized, and flattened.
5. The samples are stored in memory, grouped later by person name.

If an image cannot be read, it is skipped.

### 4. Training the Model

When the user clicks **Train Model**:

1. `FaceDatabase::Train()` checks that enough images exist.
2. It verifies all training images have the same dimensions.
3. It builds a data matrix where each column is one image vector.
4. It computes the mean face.
5. It centers the training data by subtracting the mean.
6. It computes covariance and eigenvectors.
7. It keeps the principal components that cover the top 90 percent of variance.
8. It projects every training image into feature space.
9. It builds a centroid per person.
10. It computes a rejection threshold so unknown faces can be rejected.

This is a classic eigenfaces-style pipeline.

### 5. Recognizing a New Image

When the user clicks **Upload New Image**:

1. The app checks that the model has already been trained.
2. A file picker opens.
3. The image is shown in the preview panel.
4. The same preprocessing pipeline is applied.
5. The image is projected into feature space.
6. The feature vector is compared against each person centroid.
7. The closest match is returned unless the distance is above the rejection threshold.

If the distance is too large, the result is `Unknown`.

## Important Design Choices

- The project currently uses in-memory storage only.
- Training images must all be the same size after preprocessing.
- The app is intentionally simple and does not yet crop faces before training.
- The current recognition approach is centroid-based after PCA projection.
- The code is organized so UI, model logic, and image helpers are separated.

## Build And Run

### Local build

```bash
chmod +x scripts/build.sh
./scripts/build.sh
./build/hpfrec
```

### Docker build

```bash
docker build -t hpfrec:dev .
```

### Docker run

```bash
xhost +local:root
docker run --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix hpfrec:dev
xhost -local:root
```

## Where To Start When Changing Code

If you are new to the project, read files in this order:

1. `src/app.cpp`
2. `src/main_frame.cpp`
3. `src/face_database.cpp`
4. `src/image_utils.cpp`
5. `include/hpfrec/*.hpp`

That path shows the app lifecycle from startup to UI to model training.

## Safe Extension Points

Good next changes usually live in these places:

- Add face detection or cropping in `src/image_utils.cpp` or `src/main_frame.cpp`
- Persist trained data in `src/face_database.cpp`
- Add a live camera or capture workflow in `src/main_frame.cpp`
- Add unit tests around preprocessing and recognition thresholds
