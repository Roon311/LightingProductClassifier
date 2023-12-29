# LightingProductClassifier
This Python script uses a deep learning model to classify images as either a product or an application. The classification is performed on a CSV file containing image links. The script provides an option to use multi-threading for faster processing.

## Dependencies

Make sure to install the following dependencies:

1. pandas
2. tensorflow
3. numpy
4. requests
5. sickit image
6. matplotlib
7. PySimpleGUI

You can install them using the following command:

```bash
pip install pandas tensorflow numpy requests scikit-image matplotlib PySimpleGUI
```

## Usage

1. **Change the Model Path:**
   Update the model path in the script to point to your trained deep learning model.

   ```python
   Nour4 = load_model('Model\\Nour4.h5')
   ```

2. **Run the Script:**
   Execute the script, and a GUI window will appear.
   
   ```bash
   python LightClassifier.py
   ```

3. **Select CSV File:**
   Click the "Select CSV File" button and choose a CSV file containing image links.

4. **Choose Multi-Threading (Optional):**
   Check the "Use Multi-Threading" checkbox for faster processing.

5. **Submit:**
   Click the "Submit" button to start the image classification process.

6. **View Results:**
   The classification results will be displayed on the GUI. Optionally, you can view the processing logs by clicking "View Logs."

7. **Done:**
   Click the "Done" button when the processing is complete.

## Log Viewer

If you want to review processing logs, you can click "View Logs" to open the log viewer window(currently under development).

##Application in Action

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/Roon311/LightingProductClassifier/assets/75309751/02ed3c55-d11a-40af-ac8d-6156ad81724f" style="width: 45%;">
    <img src="https://github.com/Roon311/LightingProductClassifier/assets/75309751/7630bc0f-8c02-4950-8d4e-7b9a2751b4ce" style="width: 45%;">
</div>

## Notes

- The script loads images from the provided links and classifies them using the specified deep learning model.
- If an image is classified as an application, the script searches for a product image in the CSV file and validates it using the model.
- The processed CSV file is updated with the classification results.

**Note:** Ensure that your model file, CSV file, and image links are correctly specified in the script.

Feel free to customize the script according to your requirements.
