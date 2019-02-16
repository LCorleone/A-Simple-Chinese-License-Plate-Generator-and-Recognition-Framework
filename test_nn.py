import numpy as np
import pickle
from keras.layers import Dense, Input, BatchNormalization, Conv2D, Flatten, MaxPooling2D, Activation, Reshape, Layer
from keras.models import Model, load_model, model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"]


class NormLayer(Layer):

    def __init__(self, **kwargs):
        super(NormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernal = self.add_weight(name='NormLayer', shape=(
            1, 13), initializer='ones', trainable=True)
        super(NormLayer, self).build(input_shape)

    def call(self, inputs):
        # out = self.kernal * inputs
        out = K.dot(self.kernal, inputs)
        out = K.permute_dimensions(out, (1, 0, 2))
        return out[:, 0, :]
    ''' because this NormLayer do not change the input_shape,
        so the compute_output_shape need not to implement (maybe)
    '''

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 65)

    def get_config(self):
        # config = {}
        base_config = super(NormLayer, self).get_config()
        return dict(list(base_config.items()))
        # return dict(list(base_config.items()) + list(config.items()))


e2e_model = load_model('e2e_model_v4.h5', custom_objects={'NormLayer': NormLayer})
label_path = '/home/lxy/documents/e2e_car/label.txt'
tem_label = np.loadtxt(label_path)
row, col = tem_label.shape
label1 = np.zeros([row, 31])
label2 = np.zeros([row, 34])
label3 = np.zeros([row, 34])
label4 = np.zeros([row, 34])
label5 = np.zeros([row, 34])
label6 = np.zeros([row, 34])
label7 = np.zeros([row, 34])
for i in range(row):
    label1[i, int(tem_label[i, 0])] = 1
    label2[i, int(tem_label[i, 1]) - 31] = 1
    label3[i, int(tem_label[i, 2]) - 31] = 1
    label4[i, int(tem_label[i, 3]) - 31] = 1
    label5[i, int(tem_label[i, 4]) - 31] = 1
    label6[i, int(tem_label[i, 5]) - 31] = 1
    label7[i, int(tem_label[i, 6]) - 31] = 1


img_path = '/home/lxy/documents/e2e_car/img_data.npy'
img_data = np.load(img_path)
img_data = img_data.transpose(0, 2, 1, 3)


e2e_predict = e2e_model.predict(img_data[0:6, :, :, :])



def print_trueLabel(num):
    print(np.array([np.argmax(label1[num, :]), np.argmax(label2[num, :]) + 31, np.argmax(label3[num, :]) + 31,
                    np.argmax(label4[num, :]) + 31, np.argmax(label5[num, :]) + 31, np.argmax(label6[num, :]) + 31,
                    np.argmax(label7[num, :]) + 31]))
    print(chars[np.argmax(label1[num, :])] + chars[np.argmax(label2[num, :]) + 31] + chars[np.argmax(label3[num, :]) + 31] +
          chars[np.argmax(label4[num, :]) + 31] + chars[np.argmax(label5[num, :]) + 31] + chars[np.argmax(label6[num, :]) + 31] +
          chars[np.argmax(label7[num, :]) + 31])

def print_predictLabel(x):
    num = x[0].shape[0]
    sort = np.zeros([num, len(x)])
    for i in range(len(x)):
        temp = x[i]
        sort[:, i] = np.argmax(temp, 1)
    for i in range(num):
        print(np.array([int(sort[i, 0]), int(sort[i, 1] + 31), int(sort[i, 2] + 31), int(sort[i, 3] + 31), int(sort[i, 4] + 31), int(sort[i, 5] + 31), int(sort[i, 6] + 31)]))
        print(chars[int(sort[i, 0])] + chars[int(sort[i, 1] + 31)] + chars[int(sort[i, 2] + 31)] +
              chars[int(sort[i, 3] + 31)] + chars[int(sort[i, 4] + 31)] + chars[int(sort[i, 5] + 31)] +
              chars[int(sort[i, 6] + 31)])
        


'''
def print_predictLabel(x):
    num = x.shape[1]
    for i in range(num):
        pre = x[:, i, :]
        temp = np.argmax(pre, 1)
        print(temp)
        print(chars[temp[0]] + chars[temp[1] + 31] + chars[temp[2] + 31] +
              chars[temp[3] + 31] + chars[temp[4] + 31] + chars[temp[5] + 31] +
              chars[temp[6] + 31])
'''


print_trueLabel(0)
print_trueLabel(1)
print_trueLabel(2)
print_trueLabel(3)
print_trueLabel(4)
print_trueLabel(5)
print_predictLabel(e2e_predict)

'''
def predict_label(x):
    row, col = x.shape
    sort = np.argsort(-x)
    for i in range(row):
        x[i, sort[i, 0: 7]] = 1
        x[i, sort[i, 7:]] = 0
    return x

def fastdecode(x):
    results = ""
    for i, one in enumerate(x[0]):
        if one == 1:
            results += chars[i]
    return results


out_true = fastdecode([label[2, :]])
out_pre = fastdecode(predict_label(e2e_predict))
print(out_true)
print(out_pre)
'''
'''
for i in range(out.shape[0]):
    if np.sum(label[i, :] == out[i, :]) == 65:
        predict_out[i] = 1

print(np.sum(predict_out) / 50000)
'''
