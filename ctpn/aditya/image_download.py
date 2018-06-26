#pip install chromedriver_installer
#pip install -U selenium
#pip install google_images_download

from google_images_download import google_images_download   #importing the library
response = google_images_download.googleimagesdownload()   #class instantiation
#list = ["bank cheque india","bank cheque usa","bank cheque uk","Copy of bank cheque"]

arguments = {"keywords" : "random images","limit" :400,"output_directory":"C:/Users/varin/Documents/GitHub/text-detection-ctpn/ctpn/aditya","image_directory":"random_images","print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
