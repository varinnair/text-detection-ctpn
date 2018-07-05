#pip install chromedriver_installer
#pip install -U selenium
#pip install google_images_download

import argparse

from google_images_download import google_images_download   #importing the library
response = google_images_download.googleimagesdownload()   #class instantiation
#list = ["bank cheque india","bank cheque usa","bank cheque uk","Copy of bank cheque"]

arguments = {"keywords" : "random","limit" :10,"output_directory":"C:/Users/varin/Documents/GitHub/text-detection-ctpn/classifier-docs/testing","image_directory":"random-images","print_urls":True}   #creating list of arguments
arguments['chromedriver'] = "chromedriver.exe"
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
