# faceRecognitionforRaspPi
Python scripts for Facial Recognition using EigenFaces and HaarCascades on Raspberry Pi with the PiCam
This code was cobbled together using many inspirations and open-source projects. Below they are listed.

For PiCam integration I used
  http://picamera.readthedocs.org/en/release-1.10/recipes1.html#capturing-to-an-opencv-object
  
For Initial inspiration and general structure I looked at and used
  Tony DiCola's https://learn.adafruit.com/raspberry-pi-face-recognition-treasure-box/
    

For generate.py
  I used 
  Learning OpenCV 3 Computer Vision with Python
   By Joe Minichino, Joseph Howse
   Pages 97-99 which can be found here : https://books.google.com/books?id=iNlOCwAAQBAJ&lpg=PA100&ots=iS-Ed6Wpi8&dq=opencv%20python%20working%20with%20multiple%20eigenfaces&pg=PA99#v=onepage&q&f=false 
   and also Tony DiCola's script.
   
   This will automatically generate both a .csv with your image paths and the images themselves for your training library. be sure to designate a new folder for each person. I haven't test it with multiple people just yet, so if you have issues, or discover new     things, by all means let me know.
   
For facerec.py
  I used once again, 
   Learning OpenCV 3 Computer Vision with Python
   By Joe Minichino, Joseph Howse
   Pages 97-99.
   and Tony DiCola's script along with this demo
   https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
   
Hope everyone has luck with this!
