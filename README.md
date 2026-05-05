# High-Performance Face Recognition (C++ / wxWidgets)

This project now provides a wxWidgets upload-based face recognition app backed by OpenCV PCA/eigenfaces. The UI lets you add labeled training photos, train the model, and then upload a new image to check whether it belongs to someone already in the dataset.

For a developer onboarding walkthrough, see [docs/NEW_DEVELOPER_GUIDE.md](docs/NEW_DEVELOPER_GUIDE.md).

Project layout:

- `include/hpfrec/face_database.hpp` and `src/face_database.cpp`: training, PCA/eigenfaces, and recognition
- `include/hpfrec/image_utils.hpp` and `src/image_utils.cpp`: image loading, preprocessing, and bitmap conversion
- `include/hpfrec/main_frame.hpp` and `src/main_frame.cpp`: the wxWidgets window and event handlers
- `include/hpfrec/app.hpp` and `src/app.cpp`: application startup

Quick start (build and run Docker image):

1. Build image from project root:

```bash
docker build -t hpfrec:dev .
```

2. Run the sample (container will run the built executable):

```bash
docker run --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix hpfrec:dev
```

Notes:
- The app uses file dialogs for training and recognition instead of a live camera feed.
- Training follows the classic PCA/eigenfaces pipeline: flatten images, build the data matrix, compute the mean face, center the data, derive principal components, keep the top 90% of variance, and project into feature space.
- Recognition compares a new image against the learned person centroids and rejects it as `Unknown` if it is too far from the dataset.
- The `CMakeLists.txt` finds `OpenCV` and `wxWidgets`. Optional `libtorch` support is enabled in the Docker build and installed into `/opt/libtorch`.
- For GPU or advanced ML features (PyTorch GPU, TensorFlow C++), extend the Dockerfile to fetch GPU-enabled artifacts or install CUDA toolkits.

Next steps you might want me to do:
- Add face cropping/detection before PCA training.
- Persist the trained model and dataset to disk.
- Add unit tests and CI.
