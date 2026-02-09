# tof-8x8-embedded-ml-toolchain
End-to-end toolchain for collecting 8×8 ToF sensor data, training CNN models, converting to TFLite, and deploying as C headers on ESP32-C3.

How to use:
- First you need to run the `readPlotAndSave.py` to capture the matrixes to be used for training.
- Run `makeSplits.py` to separate the data into various groups for training, validation and testing. This script also saves the background data.
- Use `checkBalance.py` to verify the data distribution between TRAIN, VAL and TEST groups.
- Run `compareCnnArhitectures.py` to evaluate some CNN architectures and decide which one is more suitable. Evalute F1 score to determine.
- Select one CNN topology and run `trainModel.py` to train a CNN based on the data provided by `makeSplits.py`. After this point a `tof_model.h5` will be available.
- Run `defineThreshold.py` to determine the detection threshold that separates what is valid and what is not.
- You can run `inference.py` if you want to check the inferece results with the data comming from the serial port.
- Run `evaluateTestSet.py` to check how your model performs agains unseen data.
- Run `exportTflite.py` to generate the Tflite models for `INT8` and `FLOAT32`.
- Execute `compareModels.py` to verify if the model quantization to `INT8` has made it loose any performance.
- Run `exportBackground.py` to generate the `background.h` to be used in the embedded code.
- Run `tfliteToHeader.py` to generate the header files `model_float.h` and `model_int8.h` to be used by the embedded code.
- Use the `background.h` and the desired model `model_float.h` or `model_int8.h` to embedded the CNN in the ESP32-C3.