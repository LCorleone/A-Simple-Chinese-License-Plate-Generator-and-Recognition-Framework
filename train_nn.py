import numpy as np
import pickle
from keras.layers import Dense, Input, BatchNormalization, Conv2D, Flatten, MaxPooling2D, Activation, Reshape, Layer, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def get_session():
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class NormLayer(Layer):

    def __init__(self, **kwargs):
        super(NormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernal = self.add_weight(name='NormLayer', shape=(
            1, 13), initializer='ones', trainable=True)
        super(NormLayer, self).build(input_shape)

    def call(self, inputs):
        # out = self.kernal * inputs
        print(inputs.shape)
        out = K.dot(self.kernal, inputs)
        out = K.permute_dimensions(out, (1, 0, 2))
        print(out.shape)
        return out[:, 0, :]
    ''' because this NormLayer do not change the input_shape,
        so the compute_output_shape need not to implement (maybe)
    '''

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        # config = {}
        base_config = super(NormLayer, self).get_config()
        return dict(list(base_config.items()))
        # return dict(list(base_config.items()) + list(config.items()))


def get_session():
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class train_e2e:

    def __init__(self):
        self.first_num = 31
        self.other_num = 34
        self.shape = (120, 30, 3)
        self.init_lr = 0.01

    def load_data(self):
        label_path = '/home/lxy/documents/e2e_car/label.txt'
        self.tem_label = np.loadtxt(label_path)
        row, col = self.tem_label.shape
        '''
        self.label = np.zeros([row, self.subject_num])
        for i in range(row):
            for j in range(col):
                self.label[i, int(self.tem_label[i, j])] = 1
        '''
        self.label1 = np.zeros([row, self.first_num])
        self.label2 = np.zeros([row, self.other_num])
        self.label3 = np.zeros([row, self.other_num])
        self.label4 = np.zeros([row, self.other_num])
        self.label5 = np.zeros([row, self.other_num])
        self.label6 = np.zeros([row, self.other_num])
        self.label7 = np.zeros([row, self.other_num])
        for i in range(row):
            self.label1[i, int(self.tem_label[i, 0])] = 1
            self.label2[i, int(self.tem_label[i, 1]) - 31] = 1
            self.label3[i, int(self.tem_label[i, 2]) - 31] = 1
            self.label4[i, int(self.tem_label[i, 3]) - 31] = 1
            self.label5[i, int(self.tem_label[i, 4]) - 31] = 1
            self.label6[i, int(self.tem_label[i, 5]) - 31] = 1
            self.label7[i, int(self.tem_label[i, 6]) - 31] = 1


        '''
        pkl_file = open(img_path, 'rb')
        U_subject = pickle.load(pkl_file)
        '''
        img_path = '/home/lxy/documents/e2e_car/img_data.npy'
        self.img_data = np.load(img_path)
        self.img_data = self.img_data.transpose(0, 2, 1, 3)

    def step_decay(self, epoch):
        if epoch % 2 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * .5)
            print("lr changed to {}".format(lr * .5))
        return K.get_value(self.model.optimizer.lr)

    def __build_network(self):
        KTF.set_session(get_session())
        input_img = Input(shape=self.shape)
        base_conv = 32
        x = input_img
        for i in range(3):
            x = Conv2D(base_conv * (2 ** (i)), (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(256, (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(1024, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        first_subject = Conv2D(self.first_num, (1, 1))(x)
        other_subject = Conv2D(self.other_num, (1, 1))(x)
        first_subject = Activation('softmax')(first_subject)
        other_subject = Activation('softmax')(other_subject)
        first_subject = Reshape((-1, self.first_num))(first_subject)
        other_subject = Reshape((-1, self.other_num))(other_subject)
        x1 = NormLayer()(first_subject)
        x2 = NormLayer()(other_subject)
        x3 = NormLayer()(other_subject)
        x4 = NormLayer()(other_subject)
        x5 = NormLayer()(other_subject)
        x6 = NormLayer()(other_subject)
        x7 = NormLayer()(other_subject)
        out1 = Activation('softmax', name='out1')(x1)
        out2 = Activation('softmax', name='out2')(x2)
        out3 = Activation('softmax', name='out3')(x3)
        out4 = Activation('softmax', name='out4')(x4)
        out5 = Activation('softmax', name='out5')(x5)
        out6 = Activation('softmax', name='out6')(x6)
        out7 = Activation('softmax', name='out7')(x7)
        rmsprop = RMSprop(lr=self.init_lr)
        self.model = Model(input_img, [out1, out2, out3, out4, out5, out6, out7])
        self.model.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 
            'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            optimizer=rmsprop, metrics=['acc'], loss_weights=[5, 1, 1, 1, 1, 1, 1])
        print(self.model.summary())

    def train(self):
        self.__build_network()
        lrate = LearningRateScheduler(self.step_decay)
        self.model.fit(self.img_data, [self.label1, self.label2, self.label3, self.label4,
            self.label5, self.label6, self.label7], epochs=20, batch_size=256, verbose=1, callbacks=[lrate], validation_split=0.1)
        self.model.save("e2e_model_v4.h5")


if __name__ == "__main__":
    my_train_nn = train_e2e()
    my_train_nn.load_data()
    my_train_nn.train()
