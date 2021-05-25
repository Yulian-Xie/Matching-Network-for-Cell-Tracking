from keras.callbacks import ModelCheckpoint
from data import get_train_val_name,generateData,generateValidData#从data.py导入数据函数  #加载训练集，验证集图片； #data for training and validation
from matplotlib import pyplot as plt
import numpy as np
import keras

# GPU
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=False   #全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
KTF.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#损失函数
def contrastive_loss(y_true, y_pred):
    tmp = y_true * tf.square(y_pred)
    tmp2 = (1 - y_true) * tf.square(tf.maximum((1 - y_pred), 0))
    return tf.reduce_sum(tmp + tmp2)/2

classes_weight = [1., 3.3, 3.2]

def train():
    EPOCHS = 500
    #导入深度网络模型
    from model import CellTracingNet
    model = CellTracingNet()
    #model_path = 'checkpoint-dir/weights-11-0.12-0.95.hdf5'
    # Load a mode

    #model.load_weights(model_path)

    # 交叉熵主要是衡量预测的0，1 概率分布和实际的0，1 值是不是匹配，交叉熵越小，说明匹配得越准确，模型精度越高
    # 平衡权重，希望哪一类平衡的更多，哪一类就小值靠前
    adam = keras.optimizers.adam(lr=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])#"binary_crossentropy"
    print(model.summary())
    #保存网络模型
    filepath ="checkpoint-dir/weights-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5"
    modelcheck = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True,mode='min',verbose=1)
    callable = [modelcheck]

    train_BS = 64
    val_BS = train_BS
    train_set, val_set = get_train_val_name()#得到训练数据
    import random
    random.shuffle(train_set)   #随机打乱
    random.shuffle(val_set)
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    H = model.fit_generator(generator=generateData(train_BS, train_set), steps_per_epoch=train_numb // train_BS,epochs=EPOCHS, verbose=1,
                            validation_data=generateValidData(val_BS, val_set), validation_steps=valid_numb // val_BS,callbacks=callable, max_q_size=1)

    #解决方案：
    #将modelll.fit([x_train, x_train], y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64)

    #改成：modelll.fit([x_train, x_train], y_train, validation_data=([x_val, x_val], y_val), epochs=10, batch_size=64)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    fig_filepath = 'train_fig'
    plt.savefig(fig_filepath)
train()