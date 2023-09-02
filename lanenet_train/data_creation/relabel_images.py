import os
import cv2
import shutil
import numpy as np

image_read_path = 'gt_image_7_10_512x256'
instance_read_path = '7_10_gt_instance_image'
image_write_path = '512x256/gt_image'
instance_write_path = '512x256/gt_instance_image'

num_processed = 0
total_images = len(os.listdir(instance_read_path))
print("Processing images...")
for file in os.listdir(instance_read_path):
    image = cv2.imread(os.path.join(instance_read_path, file))

    # exclude black!
    num_colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0).shape[0] - 1
    # remove '_' for the original dataset...
    new_filename = file[:-4] + '_' + str(num_colors) + file[-4:]
    
    shutil.copyfile(os.path.join(image_read_path, file), os.path.join(image_write_path, new_filename))
    shutil.copyfile(os.path.join(instance_read_path, file), os.path.join(instance_write_path, new_filename))

    num_processed += 1
    if num_processed % 10 == 0:
        print("Processed {}/{} images".format(num_processed, total_images))

print("Done!")