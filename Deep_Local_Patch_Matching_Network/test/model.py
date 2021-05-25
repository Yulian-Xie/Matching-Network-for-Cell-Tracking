from keras.models import Model
from keras.layers import Input,concatenate,Dense,Flatten,Dropout,GlobalAveragePooling2D,Add,MaxPooling2D
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

IMAGE_W = 128
IMAGE_H = 128
#jwb
IMAGE_C = 2

def DailtConvBlock(income,filter_num=16,wei_init='uniform',activation=LeakyReLU,Dailt_rate=1,name='ConvBlock'):

    conv = Convolution2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding="same", dilation_rate=Dailt_rate, kernel_initializer=wei_init,activation=None)(income)
    bn = BatchNormalization()(conv)
    act = activation()(bn)

    return act

def ConvBlock(income,filter_num=16,wei_init='uniform',activation=LeakyReLU,name='ConvBlock'):

    conv = Convolution2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=wei_init,activation=None)(income)
    bn = BatchNormalization()(conv)
    act = activation()(bn)

    return act


def ConvBlockDownSample(income,filter_num=16,wei_init='uniform',activation=LeakyReLU,name='ConvBlockDownSample'):

    act1 = MaxPooling2D(strides=2)(income)
    return act1



def CellTracingNet(
        input_shape=(IMAGE_W,IMAGE_H,IMAGE_C),
        output_mode="sigmoid"):
    filter_num = 8
    wei_init = 'he_normal'
    act = LeakyReLU
    # encoder
    inputs = Input(shape=input_shape)
    # inputs1
    c0_0 = ConvBlock(inputs, filter_num, wei_init=wei_init, activation=act, name='ConvBlock0_0')
    p0_0 = ConvBlockDownSample(c0_0, filter_num, wei_init=wei_init, activation=act, name='ConvBlockDownSample0')

    c1_1 = ConvBlock(p0_0, filter_num,wei_init=wei_init, activation=act,name='ConvBlock1_1')
    p1_1 = ConvBlockDownSample(c1_1, filter_num,wei_init=wei_init, activation=act,name='ConvBlockDownSample1')

    c2_1 = ConvBlock(p1_1, filter_num,wei_init=wei_init, activation=act,name='ConvBlock2_1')
    p2_1 = ConvBlockDownSample(c2_1, filter_num,wei_init=wei_init, activation=act,name='ConvBlockDownSample2')
    c2_1 = ConvBlock(p2_1, filter_num, wei_init=wei_init, activation=act, name='ConvBlock2_1')
    p2_1 = ConvBlockDownSample(c2_1, filter_num, wei_init=wei_init, activation=act, name='ConvBlockDownSample2')


    # p2_1 = DailtConvBlock(p2_1, filter_num, wei_init=wei_init, activation=act, Dailt_rate=2, name='DailtConvBlock2_1')
    p2_1 = DailtConvBlock(p2_1, filter_num, wei_init=wei_init, activation=act, Dailt_rate=3, name='DailtConvBlock2_2')
    p2_1 = DailtConvBlock(p2_1, filter_num, wei_init=wei_init, activation=act, Dailt_rate=5, name='DailtConvBlock2_3')

    # c3_1 = ConvBlock(p2_1, filter_num, wei_init=wei_init, activation=act, name='ConvBlock3_1')
    # p3_1 = ConvBlockDownSample(c3_1, filter_num, wei_init=wei_init, activation=act, name='ConvBlockDownSample3')

    fc1 = GlobalAveragePooling2D()(p2_1)
    fc2 = Dense(8, activation=None)(fc1)
    fc2 = Activation("relu")(fc2)
    fc3 = Dense(1, activation=None)(fc2)
    outputs = Activation(output_mode)(fc3)
    print("Build decoder done..")
    model = Model(inputs=inputs, outputs=outputs, name="CellTracingNet")
    # print(model.summary())

    return model


# model = CellTracingNet()