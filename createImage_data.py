import numpy as np
import cv2
import pickle

# create train label
a = np.loadtxt('label.txt')
b = np.zeros([50000, 65])
for i in range(50000):
    for j in range(7):
        b[i, int(a[i, j])] = int(a[i, j])
print(b[0, :])

# create image train data
path = 'D:/graduate/HyperLPR-master/end-to-end-for-chinese-plate-recognition-master/end-to-end-for-chinese-plate-recognition-master/data/train_data'
img_data = np.zeros([50000, 30, 120, 3])
for i in range(50000):
    img_path = path + '/' + str(i).zfill(5) + ".jpg"
    b = cv2.imread(img_path)
    img_data[i, :, :, :] = b
    print('num:' + str(i))

'''
output = open('img_data.pkl', 'wb')
pickle.dump(img_data, output)
'''
'''
img_path = path + '/' + str(0).zfill(5) + ".jpg"
b = cv2.imread(img_path)
a = np.load('pp.npy')
print(a[0,:,:,:] == b)
'''