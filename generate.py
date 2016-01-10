
print "Initializing ..."
import io
import time
import picamera
import picamera.array 
import cv2
import numpy as np
import face
import config
import csv




CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
stream = io.BytesIO()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
HAAR_FACES         = 'haarcascade_frontalface_alt.xml'
HAAR_SCALE_FACTOR  = 1.3
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (30, 30)
FACE_WIDTH  = 92
FACE_HEIGHT = 112

def doCrop(image, x, y, w, h):
	"""Crop box defined by x, y (upper left corner) and w, h (width and height)
	to an image with the same aspect ratio as the face training data.  Might
	return a smaller crop if the box is near the edge of the image.
	"""
	crop_height = int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w)
	midy = y + h/2
	y1 = max(0, midy-crop_height/2)
	y2 = min(image.shape[0]-1, midy+crop_height/2)
	return image[y1:y2, x:x+w]

def resize(image):
	"""Resize a face image to the proper size for training and detection.
	"""
	return cv2.resize(image, 
					  (FACE_WIDTH, FACE_HEIGHT), 
					  interpolation=cv2.INTER_LANCZOS4)
        
if __name__ == '__main__':
    count = 0
    filePaths = list()
    with picamera.PiCamera() as camera:
            camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
            
            
            
            while (True):
                
                with picamera.array.PiRGBArray(camera) as stream:
                    camera.capture(stream, format='bgr', use_video_port=True)
                    image = stream.array
                    
                
                
    
        
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                faces = face_cascade.detectMultiScale(image, 
                                scaleFactor=HAAR_SCALE_FACTOR, 
                                minNeighbors=HAAR_MIN_NEIGHBORS, 
                                minSize=HAAR_MIN_SIZE, 
                                flags=cv2.CASCADE_SCALE_IMAGE)

        
        
                
                        
                for (x,y,w,h) in faces:
                    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0),2)
                    crop = resize(doCrop(gray, x, y, w, h))
                    filePathForFile = ('./data/mg/%s.pgm' % str(count))
                    cv2.imwrite(filePathForFile, crop)
                    filePaths.append(filePathForFile)
                    
                    print filePaths
                    count += 1
                              
                outfile = open('mg.csv', 'w')    
                cv2.imshow('Face Image', image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    writer = csv.writer(outfile)
                    for files in filePaths:
                        writer.writerow([files,])    
                    outfile.close() 
                    break



    





                     
