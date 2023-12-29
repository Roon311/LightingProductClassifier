'''
Dependencies
1. pandas
2. tensorflow
3. numpy
4. requests
5. sickit image
6. matplotlib
7. PySimpleGUI

Change the model path
'''

import pandas as pd
import tensorflow as tf
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
#from google.colab import drive
from PIL import Image
import io
from io import BytesIO
import base64
from skimage.transform import resize
from keras.models import load_model
from tensorflow.keras.preprocessing import image

Nour4 = load_model('Model\\Nour4.h5')

base_learning_rate = 0.001
Nour4.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = base_learning_rate), loss='mean_squared_error', metrics=['accuracy'])
def detect_image_type(image_path, model):
    start_time = time.time()
    print(image_path)
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
    tempimg=img
    img = img.resize((224, 224))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = np.array(img)
    x = image.img_to_array(img)
    if x.shape[-1] < 3:
        x = np.repeat(x, 3, axis=-1)
    print(img)
    img_copy = np.copy(img)
    x = image.img_to_array(img)
    # Update the image element in the GUI
    image_drawn = tempimg.resize((200, 200))
    with io.BytesIO() as draw:
        image_drawn.save(draw, format="PNG")
        image_drawn_bytes = draw.getvalue()
    image_drawn_base64 = base64.b64encode(image_drawn_bytes).decode()
    window['_IMAGE_'].update(data=image_drawn_base64)
    x = np.expand_dims(x, axis=0)

    x = tf.keras.applications.resnet50.preprocess_input(x)

    prediction = model.predict(x)
    end_time = time.time()
    if prediction[0][0] > prediction[0][1]:
        result = "application"
    else:
        result = "product"
    print(prediction)
    detection_time = end_time - start_time
    window['_CLASSIFICATION_'].update(f'Classification: {result}')  # Update the classification label
    return result, detection_time

import threading

def classify_image(num,df,image_url, model):
    image_type, detection_time = detect_image_type(image_url, model)
    event, values = window.read(timeout=100)
    if event == 'View Logs':
            log_viewer_window.un_hide()
    if image_type== "application":
        image_path = df.iloc[num]['Additional Images & Videos']
        addlinks = image_path.split('; ')
        links_array = np.array(addlinks)
        for link in links_array:
            image_type, detection_time = detect_image_type(link, model)
            if(image_type=="product"):
               df.loc[num,'Product Image']=link 
               break

def multiThread(df,image_urls, model):
    threads = []
    start_time = time.time()
    counter=0
    for num,image_url in enumerate(image_urls):
        thread = threading.Thread(target=classify_image, args=(num,df,image_url, model))
        thread.start()
        threads.append(thread)


    for thread in threads:
        event, values = window.read(timeout=100)
        if event == 'View Logs':
            log_viewer_window.un_hide()
        counter+=1
        q.update(counter)
        thread.join()
    end_time = time.time()
    detection_time = end_time - start_time
    return detection_time

import PySimpleGUI as sg
import time

layout = [
    [
        sg.Column([
            [sg.Text('Select CSV File:'), sg.Input(key='_CSV_FILE_'), sg.FilesBrowse(file_types=(('CSV Files', '*.csv'),))],
            [sg.Checkbox('Use Multi-Threading', key='_MULTI_', default=True)],
            [sg.ProgressBar(100, orientation='h', size=(20, 20), key='_TRACE_')],
            [sg.Button('Submit'), sg.Button('Done'), sg.Button('View Logs')],
        ]),
        sg.VerticalSeparator(),
        sg.Column([
            [sg.Text('Classification:', key='_CLASSIFICATION_', size=(20, 1))],  # Classification label
            [sg.Image(key='_IMAGE_', size=(200, 200))],
        ], justification='right')
    ]
]


window = sg.Window('Model Uploader', layout)
log_viewer_layout = [
    [sg.Multiline(size=(80, 20), key='_LOGS_', autoscroll=True)],
    [sg.Button('Close')]
]
log_viewer_window = sg.Window('Log Viewer', log_viewer_layout, finalize=True)
log_viewer_window.hide() # Hide the log viewer window initially
work=False
while True:
    event, values = window.read(timeout=100)
    log_viewer_event, log_viewer_values = log_viewer_window.read(timeout=100)
    if event == sg.WIN_CLOSED :
        break
    if event=='Done':
        if work==True:
            sg.popup('The CV file is updated successfully.')
            #df.to_csv(values['_CSV_FILE_'], index=False, header=True)
            break
        else:
            sg.popup('No Processing was done.')
            break

    if event == 'View Logs':
       log_viewer_window.un_hide()

    # Check if log viewer window is visible
    if log_viewer_window and log_viewer_window.TKroot and log_viewer_window.TKroot.winfo_viewable():
        log_viewer_window['_LOGS_'].print(f'Processing completed.')

    if log_viewer_event == sg.WINDOW_CLOSED or log_viewer_event == 'Close':
        log_viewer_window.hide()

    if event == 'Submit':
        if not values['_CSV_FILE_']:
            sg.popup('Please select a CSV file.')
            continue
        selected_files = values['_CSV_FILE_']
        selected_files = selected_files.split(';')
        for file_num,file_path in enumerate(selected_files):
            df = pd.read_csv(file_path)
            product_images = df[['Product Image', 'Additional Images & Videos']].to_numpy()
            links=product_images[:, 0]
            use_multi_threading = values['_MULTI_']
            q = window['_TRACE_']
            q.update_bar(0, len(df))      
            if use_multi_threading:
                detection_time = multiThread(df,links, Nour4)
                work=True
            else:
                start_time = time.time()
                for index,image_url in enumerate(links):
                    classify_image(index,df,image_url, Nour4)
                    q.update(index+1)
                end_time = time.time()
                detection_time = end_time - start_time
                work=True
            df.to_csv(file_path, index=False, header=True)
            sg.popup(f"Detection time for file # {file_num+1} : {detection_time:.3f} s")
window.close()
log_viewer_window.close()