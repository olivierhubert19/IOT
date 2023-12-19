import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import xml.etree.ElementTree as xet
import re
import time
from glob import glob
from skimage import io
from shutil import copy
import requests

# settings
detected_plates = {}
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
# LOAD THE IMAGE

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    
    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(image,boxes_np[ind])


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

        extract_license_plate(license_text)
    return image

def extract_license_plate(license_text):
    global detected_plates
    if license_text is not None and license_text.strip():
        # Define the regular expression patterns to extract license plate formats
        pattern_1 = r'([0-9]{2}-[A-Z][0-9]+)(?:\s*\|)?(?:\s*\n)?\s*([0-9]{3}\.[0-9]{2})'
        pattern_2 = r'([0-9]{2}[A-Z]-[0-9]{3}\.[0-9]{2})'

        pattern_3 = r'([0-9]{2}[A-Z]+)(?:\s*\|)?(?:\s*\n)?\s*([0-9]{3}\.[0-9]{2})'
        
        # 29-C1 999.99
        match_1 = re.search(pattern_1, license_text, re.DOTALL)
        if match_1:
            extracted_text = f"{match_1.group(1)} {match_1.group(2)}"
            print("Extracted License Plate (Pattern 1):", extracted_text)
            
            process_license_plate(extracted_text)
        else:
            # 51F-970.22
            match_2 = re.search(pattern_2, license_text, re.DOTALL)
            if match_2:
                extracted_text = match_2.group(1)
                print("Extracted License Plate (Pattern 2):", extracted_text)
                
                process_license_plate(extracted_text)
            else :
                match_3 = re.search(pattern_3, license_text, re.DOTALL)
                if match_3:
                    extracted_text = f"{match_3.group(1)} {match_3.group(2)}"
                    print("Extracted License Plate (Pattern 3):", extracted_text)
                    
                    process_license_plate(extracted_text)
                
def process_license_plate(extracted_text):
    global detected_plates
    current_time = time.time()
    
    if extracted_text not in detected_plates or (current_time - detected_plates[extracted_text]) > 60:
        # Remove spaces before sending the API request
        cleaned_text = extracted_text.replace(" ", "")
        print(f"API request sent for: {cleaned_text}")
        # First API endpoint
        # api_endpoint_1 = f"http://localhost:8081/api/car/findcar/{cleaned_text}"
        
        # # First API request
        # response_1 = requests.get(api_endpoint_1)
        
        # # Check if the first API request was successful
        # if response_1.status_code == 200:
        #     try:
        #         json_response_1 = response_1.json()
        #         print(f"API request sent for: {cleaned_text}. Response: {json_response_1}")
                
        #     except ValueError as e:
        #         print(f"Invalid JSON response. Error: {e}. Response content: {response_1.content}")
        # else:
        #     print(f"Failed to make API request 1 for: {cleaned_text}. Status code: {response_1.status_code}")

        # Once API request is sent, update the access time in the dictionary
        detected_plates[extracted_text] = current_time
        print("Updated detected plates:", detected_plates)
    else:
        print(f"License plate {extracted_text} already processed within the 1 min, skipping API request.")
        
# predictions flow with return result
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img


# extrating text
def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'
    
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        
        return text
    
# cap = cv2.VideoCapture('./number-plate-detection/TEST/TEST.mp4')
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video')
        break

    results = yolo_predictions(frame,net)

    cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('YOLO',results)
    if cv2.waitKey(30) == 27 :
        break

cv2.destroyAllWindows()
cap.release()

