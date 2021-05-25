# coding=utf-8
import matplotlib
# from PIL import Image
matplotlib.use("Agg")
import numpy as np
import cv2
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
np.random.seed(seed)
img_w = 96
img_h = 96

DATA_PATH = 'input'
TRAIN_PATH = DATA_PATH + '/train/image1/'
LABEL_PATH = DATA_PATH + '/train/image2/'
VAL_IMG_PATH = DATA_PATH + '/val/image1/'
VAL_LAB_PATH = DATA_PATH + '/val/image2/'



def load_img(path, label=False):

    if label:#标签
        if int(path[:-4]) < 100:#正样本
            y_image = 1
        else:
            y_image = 0#负样本
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_h, img_w))

        #因为分割图像素太少，人为膨胀一次加粗线条
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel)
        img = cv2.resize(img, (img_h, img_w))

        img = img.astype(np.float32) / 255.0
        img[img > 0.15] = 1
        img[img < 0.15] = 0
        # cv2.imwrite("img3.jpg", img*255)
        img = img.astype(np.bool)
        y_image = np.zeros((128, 128), dtype=np.bool)
        y_image[16:112, 16:112] = img
    return y_image#返回图片
def load_img2(path, label=False):#验证集图片

    if label:#标签
        if int(path[:-4]) < 100:#正样本
            y_image = 1
        else:
            y_image = 0#负样本
    else:
        y_image = cv2.imread(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_h, img_w))
        # 因为分割图像素太少，人为膨胀一次加粗线条
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel)
        img = cv2.resize(img, (img_h, img_w))
        img = img.astype(np.float32) / 255.0
        img[img > 0.15] = 1
        img[img < 0.15] = 0
        # cv2.imwrite("img3.jpg", img*255)
        img = img.astype(np.bool)
        y_image = np.zeros((128, 128), dtype=np.bool)
        y_image[16:112, 16:112] = img
    return y_image#返回图片

def get_train_val_name(train_path=TRAIN_PATH,image_val_path=VAL_IMG_PATH):
    train_set = next(os.walk(train_path))[2]
    train_set.sort()
    print(train_set,len(train_set))
    val_set = next(os.walk(image_val_path))[2]
    val_set.sort()
    print(val_set,len(val_set))
    return train_set, val_set#加载训练集，验证集图片


# data for training
def generateData(batch_size, data):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1

            img1 = load_img(TRAIN_PATH + url)
            img2 = load_img(LABEL_PATH + url)
            img = np.zeros((128, 128, 2), dtype=np.bool)
            img[:, :, 0] = img1
            img[:, :, 1] = img2
            train_data.append(img)
            label = load_img(url, label=True)
            train_label.append(label)

#            cv2.imwrite("img1.jpg",img*255)
        #    cv2.imwrite("img2.jpg",img[:, :, 1]*255)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0

# data for validation
def generateValidData(batch_size, data):
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1

            img1 = load_img2(VAL_IMG_PATH + url)
            img2 = load_img2(VAL_LAB_PATH + url)
            img = np.zeros((128, 128, 2), dtype=np.bool)
            img[:, :, 0] = img1
            img[:, :, 1] = img2

            valid_data.append(img)
            #传入babel 返回标签
            label = load_img2(url, label=True)
            valid_label.append(label)

            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0

# train_set,val_set = get_train_val_name()
# import random
#
# ix = random.randint(0, len(train_set)-1)#用于生成一个指定范围内的整数
# img = load_img(TRAIN_PATH + train_set[ix])
# img = img_to_array(img)
# label = load_img(LABEL_PATH + train_set[ix])
# label = img_to_array(label)
# cv2.imwrite("img1.jpg",img*255)
# cv2.imwrite("img2.jpg",label*255)
# i = 0
