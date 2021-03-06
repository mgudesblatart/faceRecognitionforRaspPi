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
   
   This will automatically generate both a .csv with your image paths and the images themselves for your training library. Be sure to designate a new folder for each person. You may also have to make a blank .csv file before it can be used in the script. I haven't tested it with recognition of multiple people just yet, so if you have issues, or discover new things, by all means let me know.
   
For facerec.py
  I used once again, 
   Learning OpenCV 3 Computer Vision with Python
   By Joe Minichino, Joseph Howse
   Pages 100-103.
   and Tony DiCola's script along with this demo
   https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
   
Hope everyone has luck with this!

First run generate.py after editing the file to match your file structure and needs. This will create as many training files as you want. Try playing around with different haarCascades. Just change the count var to wherever you last left off the training. The more images the better.

To run the script in your command line run python facerec.py </path/to/images/>

P.S. In the images folder, should be the subfolders per individual's training images.
