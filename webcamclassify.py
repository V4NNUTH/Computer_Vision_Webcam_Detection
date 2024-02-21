import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg16
import numpy as np

image_size = 224 #image size defined as 224
model = vgg16.VGG16(weights='imagenet')
print(model.summary())

camera = cv2.VideoCapture(0) #the default camera for capturing video

while camera.isOpened():
    
    ok, cam_frame = camera.read()
    
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)#convert img to array
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())
    
    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    label = decode_predictions(predictions)#use to convert the dredict class probabilities to human-readdable label.
    
    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "VGG16: {}, {:.1f}".format(label[0][0][1],
    label[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
    cv2.imshow('video image', cam_frame)#funtion displays the video stream with the predicted class label overlayed on the image
    
    key = cv2.waitKey(30) #function waits for the user to press a key for 30 millidesonds.
    
    if key == 27: # press 'ESC' to quit
        break
    
camera.release()

cv2.destroyAllWindows() #funtion is called agter the loop exits to close any open window