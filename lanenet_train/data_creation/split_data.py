import os
from sklearn.model_selection import train_test_split

filenames = os.listdir('512x256/gt_binary_image')

X_train, X_test = train_test_split(filenames, test_size=0.2)

train_file = open('train.txt', 'a')
for filename in X_train:
    train_file.write('/home/research/brodskyd/minicity_training_data/gt_image/{} /home/research/brodskyd/minicity_training_data/gt_binary_image/{} /home/research/brodskyd/minicity_training_data/gt_instance_image/{} \n'
                     .format(filename, filename, filename))
    
train_file.close()

test_file = open('test.txt', 'a')
for filename in X_test:
    test_file.write('/home/research/brodskyd/minicity_training_data/gt_image/{} /home/research/brodskyd/minicity_training_data/gt_binary_image/{} /home/research/brodskyd/minicity_training_data/gt_instance_image/{} \n'
                     .format(filename, filename, filename))
    
test_file.close()