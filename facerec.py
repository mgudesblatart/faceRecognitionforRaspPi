print "Initializing ..."
import io
import time
import picamera
import picamera.array
import cv2
import numpy as np

import os
import sys

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

show_window = True

cam_cx = CAMERA_WIDTH/2
cam_cy = CAMERA_HEIGHT/2
inch = 9.0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
HAAR_FACES         = 'haarcascade_frontalface_alt.xml'
HAAR_SCALE_FACTOR  = 1.3
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (30, 30)

POSITIVE_THRESHOLD = 2000.0

FACE_WIDTH  = 92
FACE_HEIGHT = 112


stream = io.BytesIO()
names = ['Murray']

def read_images (path, sz=None):
        c = 0
        X,y = [], []
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    try:
                        if (filename == ".drectory"):
                            continue
                        filepath = os.path.join(subject_path, filename)
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)

                        if (sz is not None):
                                im = cv2.resize(im, sz)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(c)
                    except IOError, (errno, strerror):
                            print "I/O error({0}): {1}".format(errno,strerror)
                    except:
                            print "Unexpected error:", sys.exec_info()[0]
                            raise
                c= c+1
                                        
        return [X,y]

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

        print 'Loading training data...'
        out_dir = None
        if len(sys.argv) < 2:
                print "USAGE: facerec.py </path/to/images> [</path/to/store/images/at>]"
                sys.exit()
        
        [X,y] = read_images(sys.argv[1])
        y = np.asarray(y, dtype=np.int32)

        if len(sys.argv) == 3:
                out_dir = sys.argv[2]
        
	model = cv2.createEigenFaceRecognizer()
	model.train(np.asarray(X), np.asarray(y))
	print 'Training data loaded!'
	
        with picamera.PiCamera() as camera:
            camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)

            start_time = time.time()

            i = 0
            
            while (True):
                with picamera.array.PiRGBArray(camera) as stream:
                    camera.capture(stream, format='bgr', use_video_port=True)
                    
                    image = stream.array

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                
                faces = face_cascade.detectMultiScale(image, 
                                scaleFactor=HAAR_SCALE_FACTOR, 
                                minNeighbors=HAAR_MIN_NEIGHBORS, 
                                minSize=HAAR_MIN_SIZE, 
                                flags=cv2.CASCADE_SCALE_IMAGE)

                    
                
                for (x,y,w,h) in faces:

                        
                        print 'X = %s, Y = %s, W = %s, H = %s ' % (x, y, w, h)
                        start_time = time.time()
                        image2 = gray
                        crop = resize(doCrop(image2, x, y, w, h))
                        #cv2.imshow('Crop', crop)
                        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0),2)
                        # Test face against model.
                        params = model.predict(crop)
                        
                        label, confidence = (params[0], params[1])
                        print "Label: %s, Confidence: %.2f" % (label, confidence)
                        if confidence < POSITIVE_THRESHOLD:
                                print 'Recognized face!'
                                cv2.putText(image, names[params[0]], (x, y + h + 20),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
                                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)
                        else:
                                print 'Did not recognize face!'
                                cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255),2)

                            
                    
                elapsed_time = time.time() - start_time

                
                cv2.imshow('Face Image', image)
                #cv2.imshow('Gray Image', gray)
                #stream.truncate(0)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
