# import the necessary packages
from playsound import playsound
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import tkinter
from tkinter import messagebox
import smtplib
import os
import serial

arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

ard='0'
sec=0
yog=0
email_id = os.environ.get('email_user1')
email_passw = os.environ.get('email_pass1')
email_iddd = os.environ.get('email_user')
filename= 'video.avi'
frames_per_seconds= 100.0

def write_read(x):
    arduino.write(bytes(x,'utf-8'))
    time.sleep(0.05)
def detect_and_predict_mask(frame, faceNet, maskNet):
# grab the dimensions of the frame and then construct a blob
# from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0)) #blob is used to detect the bright features from background
    data = arduino.readline()
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)   # FaceNet is  a deep neural network used for extracting features from an image of a persons face
    detections = faceNet.forward() #Runs a forward pass to compute the net output.
    #print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

# loop over the detections
    for i in range(0, detections.shape[2]):
# extract the confidence (i.e., probability) associated with
# the detection
        confidence = detections[0, 0, i, 2]

# filter out weak detections by ensuring the confidence is
# greater than the minimum confidence
        if confidence > 0.5:
# compute the (x, y)-coordinates of the bounding box for
# the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

# ensure the bounding boxes fall within the dimensions of
# the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

# extract the face ROI, convert it from BGR to RGB channel
# ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

# add the face and bounding boxes to their respective
# lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

# only make a predictions if at least one face was detected
    if len(faces) > 0:
# for faster inference we'll make batch predictions on all
# faces at the same time rather than one-by-one predictions
# in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

# return a 2-tuple of the face locations and their corresponding
# locations
    return (locs, preds)

#print(email_id,email_passw)
#print(email_iddd)
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"  #The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel" #The .caffemodel file which contains the weights for the actual layers
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)  #Both files are required when using models trained using Caffe for deep learning.

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

#dims=get_dims(vs, res=my_res)
#video_type_cv2=get_video_type(filename)
#out = cv2.VideoWriter(filename, video_type_cv2, frames_per_seconds, dims)

# loop over the frames from the video stream
while True:
# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
    yog+=1
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
#out.write(frame)
# detect faces in the frame and determine if they are wearing a
# face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

# loop over the detected face locations and their corresponding
# locations
    for (box, pred) in zip(locs, preds):
# unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
# determine the class label and color we'll use to draw
# the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        if label=="No Mask" and (max(mask, withoutMask) * 100)>95:
            ard="1"
            sec=sec+1
            if sec>=25:
                sec=0
                print("Sending mail")
                with smtplib.SMTP('smtp.gmail.com',587) as smtp:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.ehlo()
                    smtp.login(email_id, email_passw)
                    subject = 'Start Face Mask Surveillance'
                    body = 'Please ensure that COVID protocols are being followed'
                    msg= f'Subject: {subject}\n\n{body}'
                    smtp.sendmail(email_id,email_iddd,msg)
                print("mail sent successfully")
            playsound("C:/Mask Detection/CODE/Face-Mask-Detection-master/play.mp3",False)
        if label=="Mask" and (max(mask, withoutMask) * 100)>95:
            ard="0"
        write_read(ard)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
# include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
#print(yog,max(mask, withoutMask) * 100,end=" " )
#print()
# display the label and bounding box rectangle on the output
# frame
        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
#cv2.release()
#out.release()
print(yog)
cv2.destroyAllWindows()
vs.stop()
