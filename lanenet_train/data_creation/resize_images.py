import os
import shutil
import cv2

read_path = '../data/orig/image_1280x170'
write_path = '../data/orig/image_512x256'

if os.path.exists(write_path):
    shutil.rmtree(write_path)

os.mkdir(write_path)

print("Processing images...")
for file in os.listdir(read_path):
    if file == '.DS_Store':
	    continue

    image = cv2.imread(os.path.join(read_path, file))
    image = cv2.resize(image, (512, 256), cv2.INTER_AREA)
    cv2.imwrite(os.path.join(write_path, file), image)

print("Done!")