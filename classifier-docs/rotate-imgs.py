# this script rotates images clockwise and anti clockwise
import glob
import cv2
import imutils
import os
import math

# function to rotate images without cropping
def rotate_image(mat, angle):
  # angle in degrees
  height, width = mat.shape[:2]
  image_center = (width/2, height/2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

path = "C:/Users/varin/Documents/GitHub/text-detection-ctpn/classifier-docs/training/all-bank-cheques"

jpg = "*.jpg"
png = "*.png"
jpeg = "*.jpeg"
tif = "*.tif"

i = 0
# iterating through extensions
for img_type in [jpg, png, jpeg, tif]:
    
    # reading all imgs of a particular extension
    images = [cv2.imread(file) for file in glob.glob(os.path.join(path, img_type))]

    # iterating through each image
    j = 0
    for image in images:
        
        # rotating img 90 degrees anti-clockwise
        #anti_clockwise_rotated = imutils.rotate(image, 90)
        anti_clockwise_rotated = rotate_image(image, 90)

        # rotating img 90 degrees clockwise, which is the same as 270 degrees anti-clockwise
        #clockwise_rotated = imutils.rotate(image, 270)
        clockwise_rotated = rotate_image(image, 270)

        # checking img file extension
        """extension = ""
        if img_type == jpg:
            extension = ".jpg"
        elif img_type == png:
            extension == ".png"
        else:
            extension == ".jpeg" """

        # saving rotated imgs
        cv2.imwrite(os.path.join(path, str(i)+str(j)+"anticlockwise.jpg"), anti_clockwise_rotated)
        cv2.imwrite(os.path.join(path, str(i)+str(j)+"clockwise.jpg"), clockwise_rotated)

        j += 1
    i += 1
