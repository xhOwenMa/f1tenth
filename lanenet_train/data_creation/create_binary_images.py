import cv2
import os
import shutil

read_path = '512x256/gt_instance_image'
write_path = '512x256/gt_binary_image'

if os.path.exists(write_path):
    shutil.rmtree(write_path)
os.mkdir(write_path)


print("Processing images...")
for file in os.listdir(read_path):
    _, binary_image = cv2.threshold(cv2.imread(os.path.join(read_path, file), cv2.IMREAD_GRAYSCALE), 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(write_path, file), binary_image)
                
print("Done!")