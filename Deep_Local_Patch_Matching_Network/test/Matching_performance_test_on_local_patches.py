# 128*128 先循环上一帧细胞，再用循环下一帧细胞，细胞块与细胞块匹配
# -*- encoding: utf-8 -*-
'''
@File     :   py
@Time     :   2021/05/10 10:09:54
@Author   :   Y.L.Xie
@Function :   网络二分类性能测试，直接用标签：t_label=[]，网络计算的相似度：t_pre=[]生成fpr，tpr，t，导出fpr，tpr，t文件
@fix:
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
from test.model import CellTracingNet
import numpy as np
import cv2
from tqdm import tqdm


IMAGE_W = 96
IMAGE_H = 96
resize_W = 96
resize_H = 96

IMAGE_C = 2

# Load a model
model = CellTracingNet()#加载网络模型
model_path = '../train/checkpoint-dir/weights-125-0.004-0.951.hdf5'#加载训练好的参数
# Load a mode
model.load_weights(model_path)

#11细胞块、坐标文件
DATA_PATH = './crop_patches/'
#存放 标号后的细胞大图
test=2


# Get and resize train images and masks，对测试集图片进行处理 加粗线条
def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (resize_H,resize_W))
    # 因为分割图像素太少，人为膨胀一次加粗线条
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.dilate(img, kernel)
    img = cv2.resize(img, (resize_W, resize_H))

    img = img.astype(np.float32) / 255.0
    img[img > 0.15] = 1
    img[img < 0.15] = 0
    img = img.astype(np.bool)
    y_image = np.zeros((128, 128), dtype=np.bool)
    y_image[16:112, 16:112] = img
    return y_image  #返回图片

#加载细胞块图
def load_test_data(filename):
    TEST_PATH = DATA_PATH + 'crop_result/'
    # 获取细胞块
    path1 = TEST_PATH + filename[:-4] + '_1/'#得到第一张图的每一个细胞块  000001.bmp/
    image_ids1 = next(os.walk(path1))[2]
    image1 = []
    for i, name in tqdm(enumerate(image_ids1), total=len(image_ids1)):#索引从0开始
        temp = path1 + name    #name:000001.bmp
        img = load_image(temp)  #加粗线条
        image1.append(img)

    path2 = TEST_PATH + filename[:-4] + '_2/'#得到第二张图的每一个细胞块
    image_ids2 = next(os.walk(path2))[2]
    image2 = []
    for i, name in tqdm(enumerate(image_ids2), total=len(image_ids2)):
        temp = path2 + name #name 图片名称
        img = load_image(temp)
        image2.append(img)

    return image1,image2        #返回列表


# 加载模型
import keras
adam = keras.optimizers.adam(lr=0.028)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
print(model.summary())

#读取所有细胞大图
TEST_SRC1 = DATA_PATH + 'segementation1/'
TEST_SRC2 = DATA_PATH + 'segementation2/'
# 测试图片
positive_num=[0]*100
passive_num=[0]*100

t_label=[]
t_pre=[]

image_ids = next(os.walk(TEST_SRC1))[2]

for l in range(1):
    score_t=np.float32(0.01*(l+1)) #二分类阈值
    print("score:",score_t,type(score_t))

    for i, name in tqdm(enumerate(image_ids), total=len(image_ids)):#进度条 enumerate索引从0开始 name:000003 、7、8  每一张大图
        #  加载大图
        if i==0:
            src1 = cv2.imread(TEST_SRC1+ name)#segementation1/000003.jpg
            src2 = cv2.imread(TEST_SRC2 + name)#segementation2/000003.jpg
            # 加载多张细胞块图
            image1, image2 = load_test_data(name) #crop_result/000003_1 、crop_result/000003_2
            # 加载中心细胞坐标
            num = []
            for k in range(0, len(image2)):  # crop_result/..._2的每个细胞块
                num.append(k+1)
            # print(num)
            right_num=0
            s = 1
            for j in range(len(image1)):#crop_result/..._1的每个细胞块

                # 存储排序后的匹配度列表
                max_list = []
                # 设立字典，存储j，k，为中心细胞的索引
                score = {}
                # 存储匹配度最大值
                max = 0
                p=0

                for k in range(0,len(image2)):#crop_result/..._2的每个细胞块
                    # 重合中心细胞，融合图像
                    if k==j:
                        y_pred_image = np.zeros((128, 128, 2), dtype=np.bool)
                        y_pred_image[:, :, 0] = image1[j]   #crop_result/000003_1
                        y_pred_image[:, :, 1] = image2[k]   #crop_result/000003_2
                        y_pred_image = np.expand_dims(y_pred_image, axis=0)  #扩展0通道
                        y_pred = model.predict(y_pred_image)  #预测的匹配度
                        temp = y_pred[0][0]
                        print(type(temp))
                        t_label.append(1)
                        t_pre.append(temp)
                        if float(temp) >= score_t:
                            right_num = right_num + 1
                            print('正样本预测值大于阈值的情况:', temp, "上一帧细胞序号:", j + 1, "下一帧细胞序号：", k+1)

            positive_num[l]=right_num

        if i == 1:
            src1 = cv2.imread(TEST_SRC1 + name)  # segementation1/000003.jpg
            src2 = cv2.imread(TEST_SRC2 + name)  # segementation2/000003.jpg
            # 加载多张细胞块图
            image1, image2 = load_test_data(name)  # crop_result/000003_1 、crop_result/000003_2
            # 加载中心细胞坐标
            num = []
            for k in range(0, len(image2)):  # crop_result/..._2的每个细胞块
                num.append(k + 1)
            # print(num)
            right_num = 0
            s = 1
            for j in range(len(image1)):  # crop_result/..._1的每个细胞块

                # 存储排序后的匹配度列表
                max_list = []
                # 设立字典，存储j，k，为中心细胞的索引
                score = {}
                # 存储匹配度最大值
                max = 0
                p = 0

                for k in range(0, len(image2)):  # crop_result/..._2的每个细胞块
                    # 重合中心细胞，融合图像
                    if k == j:
                        y_pred_image = np.zeros((128, 128, 2), dtype=np.bool)
                        y_pred_image[:, :, 0] = image1[j]  # crop_result/000003_1
                        y_pred_image[:, :, 1] = image2[k]  # crop_result/000003_2
                        y_pred_image = np.expand_dims(y_pred_image, axis=0)  # 扩展0通道
                        y_pred = model.predict(y_pred_image)  # 预测的匹配度
                        temp = y_pred[0][0]
                        t_label.append(0)
                        t_pre.append(temp)
                        if temp < score_t:
                            right_num = right_num + 1
                            print('负样本预测值小于阈值的情况:', temp, "上一帧细胞序号:", j + 1, "下一帧细胞序号：", k + 1)

            passive_num[l] = right_num

print(positive_num)#正->正的个数
print(passive_num)#负->负的个数

# for m in range(len(t_label)):
#     print(m)
#     print("t_label",t_label[m])
#     print("t_pre",t_pre[m])
#     print("-----")
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_label = t_label
y_pre = t_pre
fpr, tpr, thersholds = roc_curve(y_label, y_pre)


roc_auc = auc(fpr, tpr)

roc_path = 'DLPN-matlab_crop54.txt'

f = open(roc_path,'a')# 读取label.txt文件，没有则创建，‘a’表示再次写入时不覆盖之前的内容
f.write('fpr')
f.write(',')
f.write('tpr')
f.write(',')
f.write('t')
f.write('\n')

for i, value in enumerate(thersholds):
    print("%f %f %f" % (fpr[i], tpr[i], value))
    f = open(roc_path, 'a')  # 读取label.txt文件，没有则创建，‘a’表示再次写入时不覆盖之前的内容
    f.write( str(fpr[i]) )
    f.write(',')  # 实现缩进
    f.write( str(tpr[i]) )
    f.write(',')  # 实现缩进
    f.write( str(value) )
    f.write('\n')  # 实现换行的功能

plt.plot(fpr, tpr, color='darkorange',label='ROC (area = {0:.3f})'.format(roc_auc), lw=2)
# plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.3f})'.format(roc_auc), lw=2)

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thersholds)
print(optimal_th)


