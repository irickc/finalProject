# Dependencies
Necessary dependencies are listed in the comments in the notebook. Additional dependencies for running the demo are included as comments in the code cells for the demos.

# Data and Training
The datasets used for training and testing are included as 7zip files in the `/Datasets` directory. These must be extracted and the `DATASET_DIR` variable located in the configuration cell of the notebooks must point to the directory where the extracted datasets are located. By default, `DATASET_DIR` points to `./Datasets`.

### SVM Specifics
Running all of the cells before the demo in `SVM-PCA.ipynb` will train and test the model. The `predict()` function can instead be used to perform testing based on a pretrained model with the artifacts located in the `/outputs` directory.

### CNN Specifics
The CNN notebook includes a `DO_TRAINING` flag which will train the model when running the `main()` function and set to `True`, or load the pre-trained model from the artifacts in `/cnnModels` and perform testing when set to `False`. Additionally, PyTorch will use the GPU if available.

# Outputs
The output directory can be configured in the `OUTPUT_DIR` variable of the configuration cell of both notebooks. You may need to additionally create a `CNN` and an `SVM` directory inside of the `OUTPUT_DIR`.