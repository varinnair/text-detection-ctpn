#pip install chromedriver_installer
#pip install -U selenium
#pip install google_images_download

from google_images_download import google_images_download   #importing the library
response = google_images_download.googleimagesdownload()   #class instantiation
list = ["bank cheque india","bank cheque usa","bank cheque uk","Copy of bank cheque"]

for i in list:
    arguments = {"keywords" : i,"limit" :100,"output_directory":"/home/aditya/research/dum1","image_directory":i,"print_urls":True}   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function
    print(paths)   #printing absolute paths of the downloaded images












