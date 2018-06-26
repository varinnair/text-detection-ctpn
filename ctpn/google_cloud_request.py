import io
import os
from google.cloud import vision
from google.cloud.vision import types
from lib.fast_rcnn.config import cfg

# Wrote the code using this: https://www.youtube.com/watch?v=tqFk8bzv2ys

def image_to_text(image_path):

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.ROOT_DIR+'/ctpn/apikey.json'

    client = vision.ImageAnnotatorClient()

    file_name = image_path

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs text detection on the image file
    # Since we want the text and not the labels, we change the method to 
    # client.text_detection(image=image) instead
    response = client.text_detection(image=image)

    # getting text annotations
    text = response.text_annotations

    if(len(text) >0):
        final_text = text[0].description
    else:
        final_text = ""
        
    print(final_text)

    return final_text


#image_to_text('C:/Users/varin/Documents/GitHub/text-detection-ctpn/data/img1/subsection_10.jpg')
