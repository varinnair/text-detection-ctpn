from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

from PIL import Image
import pytesseract
import argparse

from google_cloud_request import image_to_text

# creating new directory for the uploaded cheque image
# the directory stores original image uploaded, processed image, subsection images, and text files
# containing coordinates
# for each cheque uploaded, we have one directory
global dir_name # this directory name will be determined by name of picture uploaded
global list_of_texts

class tesseract():
    def __init__(self,path):
        self.path = path
    def change_path(self,new_path):
        self.path = new_path
    def get_path(self):
        return self.path
    def process(self):

        image = cv2.imread(self.path)
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, image)
        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
        return text

def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

# Don't really need this. We already have the text files with the coordinates
def draw_boxes(img,image_name,boxes,scale):
    dir_name = "img1"
    base_name = image_name.split('\\')[-1]
    #have to change the line below
    with open('data/'+dir_name+'/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
            f.write(line)

    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/"+dir_name, "boxed_"+base_name), img) #have to change

def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

# convert_subsection_to_text method uses tesseract - not as accurate as Google Cloud Vision API
def convert_subsection_to_text(subsection_file_path):
    class1 = tesseract(subsection_file_path)
    return class1.process()

# helper function for refine_text
def is_alphanumeric_or_space(c):
    x = ord(c)
    if (x >= 65 and x <= 90) or (x >= 97 and x <= 122) or (x >= 48 and x <= 57) or c == " ":
        return True
    return False

# removes undefined characters from string
def refine_text(text):
    final_string = ""
    for c in text:
        if is_alphanumeric_or_space(c):
            final_string += c

    return final_string

# creates subsections from the coordinates returned by CTPN
def create_image_subsections():
    dir_name = "img1"
    img = cv2.imread('data/'+dir_name+'/'+dir_name+'.jpg') # opening cheque image
    gray = img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting image to gray scale
    #gray = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #setting threshold

    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)
    
    lines = [line.rstrip('\n') for line in open('data/'+dir_name+'/res_img1.txt')] #opening text file containing coordinates for boxes

    coords = []
    #adding all the coordinates in a list of lists called coords, which holds coordinates for all the boxes
    for line in lines:
        line = line.strip()
        line.replace("\n", "")
        nums = line.split(',')
        coords.append(nums)

    #iterating through each coordinate
    i = 0
    while i < len(coords):
        dimensions = coords[i] #list of 4 elements in the order => y1, x1, y2, x2
    
        y1 = int(dimensions[0].strip())
        y2 = int(dimensions[2].strip())

        x1 = int(dimensions[1].strip())
        x2 = int(dimensions[3].strip())
    
        #finding image subsection
        img_subsection = gray[x1:x2, y1:y2]
        subsection_file_name = 'subsection_'+str(i)+'.jpg'

        #saving image subsection
        subsection_file_path = 'data/'+dir_name+'/'+subsection_file_name
        cv2.imwrite(subsection_file_path, img_subsection)
        subsection_file_path = cfg.ROOT_DIR+'/'+subsection_file_path

        #converting the subsection to text using Google Cloud Vision API
        #text = image_to_text(subsection_file_path)

        # converting the subsection to text using Tesseract
        #text = convert_subsection_to_text(subsection_file_path)

        #text = refine_text(text)
        #list_of_texts.append(text)
        
        i += 2

if __name__ == '__main__':

    # TODO
    dir_name = "img1" #need to get dir_name from webApp
    #os.makedirs("data/" + dir_name + "/")
    # TODO put the image in the directory

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, dir_name, '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, dir_name, '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)
    
    list_of_texts = []
    create_image_subsections()

    #now, list_of_texts contains all the strings
    print(list_of_texts)
