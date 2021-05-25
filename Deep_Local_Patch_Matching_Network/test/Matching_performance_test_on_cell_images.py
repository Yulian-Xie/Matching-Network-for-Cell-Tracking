# 128*128 细胞两两匹配 相同细胞相同的标号 用km算法进行最优匹配，进行排序，权重最大为第一对匹配细胞
#模型：model.py
#
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

from test.model import CellTracingNet
import numpy as np
from scipy import ndimage
import cv2
from tqdm import tqdm
import os
from test.km_matcher import KMMatcher

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
DATA_PATH = './crop_images/'


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
    path1 = TEST_PATH + str(int(filename[:-4])) + '_1/'#得到第一张图的每一个细胞块  000001.bmp/
    image_ids1 = next(os.walk(path1))[2]
    image1 = []
    for i, name in tqdm(enumerate(image_ids1), total=len(image_ids1)):#索引从0开始
        temp = path1 + name    #name:000001.bmp
        img = load_image(temp)  #加粗线条
        image1.append(img)

    path2 = TEST_PATH + str(int(filename[:-4]))  + '_2/'#得到第二张图的每一个细胞块
    image_ids2 = next(os.walk(path2))[2]
    image2 = []
    for i, name in tqdm(enumerate(image_ids2), total=len(image_ids2)):
        temp = path2 + name #name 图片名称
        img = load_image(temp)
        image2.append(img)

    return image1,image2        #返回列表

# 获取细胞块坐标，当匹配成功时，会在原图中编号
def read_txt_coordinate(name):

    TEST_TXT1 = DATA_PATH + 'segement_txt1/'#单细胞坐标
    TEST_TXT2 = DATA_PATH + 'segement_txt2/'

    file = open(TEST_TXT1 +str(int(name[:-4]))  + '.txt', "r")
    list = file.readlines() #读取所有行
    txt1 = []
    for i, fields in enumerate(list, len(list)): #索引从len(list)开始  #field 数据：119.7445,257.1690,

        fields = fields.strip()  # s.strip(rm)  删除s字符串中开头、结尾处，位于 rm删除序列的字符
        fields = fields.strip("[]")
        fields = fields.replace('\t', ',')#“,”替代“\t”
        fields = fields.split(',', 1)  # 用“,”分割坐标数据 ：
        txt1.append(fields)

    file = open(TEST_TXT2 + str(int(name[:-4]))  +'_1'+ '.txt', "r")
    list = file.readlines()
    txt2 = []
    for i, fields in enumerate(list, len(list)):
        fields = fields.strip()  # s.strip(rm)  删除s字符串中开头、结尾处，位于 rm删除序列的字符
        fields = fields.strip("[]")
        fields = fields.replace('\t', ',')
        fields = fields.split(',', 1)  # 分割坐标数据 ：
        txt2.append(fields)

    # file = open(TEST_TXT2 + name[:-4] + '_2'+ '.txt', "r")
    # list = file.readlines()
    # txt3 = []
    # for i, fields in enumerate(list, len(list)):
    #     fields = fields.strip()  # s.strip(rm)  删除s字符串中开头、结尾处，位于 rm删除序列的字符
    #     fields = fields.strip("[]")
    #     fields = fields.replace('\t', ',')
    #     fields = fields.split(',', 1)  # 分割坐标数据 ：
    #     txt3.append(fields)

    return txt1, txt2

def cell_num_txt1(name):
    TEST_TXT1 = DATA_PATH + 'cell_num_txt1/'  #
    file = open(TEST_TXT1 + str(int(name[:-4])) + '_1' + '.txt', "r")
    list1 = file.readlines()  # 读取所有行
    return list1

def cell_num_txt2(name):
    TEST_TXT2 = DATA_PATH + 'cell_num_txt2/'  #
    file = open(TEST_TXT2 +str(int(name[:-4]))  + '_2' + '.txt', "r")
    list2 = file.readlines()  # 读取所有行
    return list2

# 加载模型
import keras
adam = keras.optimizers.adam(lr=0.028)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
print(model.summary())

#读取所有细胞大图
TEST_SRC1 = DATA_PATH + 'segementation1/'
TEST_SRC2 = DATA_PATH + 'segementation2/'
# 测试图片
image_ids = next(os.walk(TEST_SRC1))[2]

number = []
for k in range(0, len(image_ids)+1):  # crop_result/..._2的每个细胞块
    number.append(0)
match_number=0

print(number)
for i, name in tqdm(enumerate(image_ids), total=len(image_ids)):#进度条 enumerate索引从0开始 name:34 35 36 37 38.bmp  每一张大图
    #  加载大图
    src1 = cv2.imread(TEST_SRC1+ name)#segementation1/000003.jpg
    src2 = cv2.imread(TEST_SRC2 + name)#segementation2/000003.jpg
    print(src1.shape)
    # 加载多张细胞块图
    image1, image2 = load_test_data(name) #crop_result/000003_1 、crop_result/000003_2
    # 加载中心细胞坐标
    txt1, txt2= read_txt_coordinate(name)#segement_txt1/000003.txt 、segement_txt2/000003_1.txt、segement_txt2/000003_2.txt
    #记录下一帧未匹配细胞块所用列表
    num1 = []
    for k in range(0, len(image1)):  # crop_result/..._2的每个细胞块
        num1.append(k+1)

    num2 = []
    for l in range(0, len(image2)):  # crop_result/..._2的每个细胞块
        num2.append(l+1)

    num3 = []
    for m in range(0, len(image1)):  # crop_result/..._2的每个细胞块
        num3.append(m + 1)

    num4 = []
    for n in range(0, len(image2)):  # crop_result/..._2的每个细胞块
        num4.append(n + 1)
    # print(num3)
    # print(num4)
    num5 = num3
    num6 = num4
    #细胞块的数量
    a=len(image1)
    b=len(image2)
    #写入匹配的细胞序号到tracklet 创建文件夹
    # node = DATA_PATH + 'tracklet/' + 'node' + str(name[:-4]) + '/'
    # if not os.path.exists(node):
    #     os.makedirs(node)  # 创建新文件夹

    if a<b:  #上一帧和下一帧匹配
        s = 1 #匹配数量
        score_list= []
        distance_list=[]
        score = {}

        for j in range(len(image1)):#crop_result/xxx_1的每个细胞块
            src1_coordinate = txt1[j]
            # 中心细胞坐标
            x1 = float(src1_coordinate[0])
            y1 = float(src1_coordinate[1])

            # 存储排序后的匹配度列表
            max_list = []

            # 存储匹配度最大值
            max = 0

            for k in range(0,len(image2)):#crop_result/xxx_2的每个细胞块
                # 重合中心细胞，融合图像
                src2_coordinate = txt2[k]
                # 第二张图的每个细胞块
                x2 = float(src2_coordinate[0])
                y2 = float(src2_coordinate[1])
                w = x1 - x2 #两个细胞位置的差距
                h = y1 - y2
                import math
                # distance = 10 / math.sqrt(math.pow(w, 2) + math.pow(h, 2))

                y_pred_image = np.zeros((128, 128, 2), dtype=np.bool)
                y_pred_image[:, :, 0] = image1[j]   #crop_result/000003_1
                y_pred_image[:, :, 1] = image2[k]   #crop_result/000003_2
                y_pred_image = np.expand_dims(y_pred_image, axis=0)  #扩展0通道
                y_pred = model.predict(y_pred_image)  #网络预测的匹配度
                temp = y_pred[0][0]
                # if(j+1==18):
                #     print("单细胞{0}与双细胞{1}的预测值为{2}".format(j+1,k+1,temp))#查看分裂的细胞的预测值
                #
                # if max < temp:
                #     max = temp
                # if temp > 0.1:
                #         print('准确率大于0.5的匹配情况:', temp, "单细胞序号:", num1[j], "组合细胞序号：", num2[k])
                score_list.append(temp)#列表，保存准确率，一维矩阵
                # distance_list.append(distance)#保存 1/距离
                # print("列表1的维度:", np.array(score_list).shape)

                dict = '%s'%temp
                score[dict] = [j, k, num1[j], num2[k]]#j,k, j+1, num[k] j k 程序中的序号， j + 1, num[k] 实际的文件名 >=1
              #  if temp>0.80:
              #      cv2.imwrite("Debug/%.6d_1.jpg" % k, image1[j] * 255)
              #      cv2.imwrite("Debug/%.6d_2.jpg" % k, image2[k] * 255)
        score_list1 = np.array(score_list).reshape(a, b) #列表，保存准确率，转换为二维矩阵
        # distance_list = np.array(distance_list).reshape(a, b)
        # score_list2 = np.multiply(score_list1, distance_list)

        print("图:",name[:-4],"相似度矩阵维度:{}", np.array(score_list1).shape)

        matcher = KMMatcher(score_list1)   #km算法
        best, weight = matcher.solve(verbose=True) #km算法结果 返回一个字典

        weight = np.array(weight)  # 对匹配度降序排列
        weight = -weight
        weight_sort = weight.argsort()

        s=1
        for k in range(0, a):  #km算法匹配为a对
            max_score = best.get(str(weight_sort[k]))  # 匹配值最大的对应细胞的序号和权重
            # print(max_score[2])

            if max_score[2]> 0.5:
                print("大图{0},find:{1},上一张{2}号和下一张{3}号细胞 权重：{4}".format(name[:-4], s, int(max_score[0]), int(max_score[1]),max_score[2]))

                src1_coordinate = txt1[int(max_score[0])-1]  # 第一张图对应的细胞序号的中心细胞的坐标
                x1 = float(src1_coordinate[0])  # 横坐标
                y1 = float(src1_coordinate[1])  # 纵坐标
                # print(src1_coordinate)
                # 上一帧原图编号
                # j = int(max_score[0] - 1)  # 上一帧的序号 j
                # k = int(max_score[1] - 1)  # 下一帧匹配的序号 num[k]-1
                cell_mark_num = s  # 给一对匹配大图标号 j位置的数字
                src1 = cv2.putText(src1, '%d' % cell_mark_num, (int(x1 - 5), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                   (0, 255, 0), 1)  # 标号
                cv2.imwrite('result/' + name[:-4] + '_1.jpg', src1)  # 在原图上标号
                src2_coordinate = txt2[int(max_score[1])-1]  # 第二张图对应的细胞1坐标
                x2 = float(src2_coordinate[0])
                y2 = float(src2_coordinate[1])
                src2 = cv2.putText(src2, '%d' % cell_mark_num, (int(x2 - 5), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                   (0, 255, 0), 1)
                cv2.imwrite('result/' + name[:-4] + '_2.jpg', src2)

                # print("num3:",int(max_score[0]))
                # print("num4:",int(max_score[1] - 1))
                s += 1
                del num3[num3.index(int(max_score[0]))]  # 删除上一帧匹配上的
                del num4[num4.index(int(max_score[1]))]  # 删除下一帧匹配上的


    else:#
        s = 1
        score_list = []
        distance_list = []
        score = {}
        for j in range(len(image2)):  # crop_result/..._1的每个细胞块

            src1_coordinate = txt2[j]
            # 中心细胞坐标
            x1 = float(src1_coordinate[0])
            y1 = float(src1_coordinate[1])

            # 存储排序后的匹配度列表
            max_list = []
            # 设立字典，存储j，k，为中心细胞的索引

            # 存储匹配度最大值
            max = 0

            for k in range(0, len(image1)):  # crop_result/..._2的每个细胞块
                # 重合中心细胞，融合图像
                src2_coordinate = txt1[k]
                # 第二张图的每个细胞块
                x2 = float(src2_coordinate[0])
                y2 = float(src2_coordinate[1])
                w = x1 - x2  # 两个细胞位置的差距
                h = y1 - y2
                import math
                # distance = 10/math.sqrt(math.pow(w,2)+math.pow(h,2))

                y_pred_image = np.zeros((128, 128, 2), dtype=np.bool)
                y_pred_image[:, :, 0] = image2[j]  # crop_result/000003_1
                y_pred_image[:, :, 1] = image1[k]  # crop_result/000003_2
                y_pred_image = np.expand_dims(y_pred_image, axis=0)  # 扩展0通道
                y_pred = model.predict(y_pred_image)  # 预测的匹配度

                temp = y_pred[0][0]

                # if(j+1==18):
                #     print("单细胞{0}与双细胞{1}的预测值为{2}".format(j+1,k+1,temp))#查看分裂的细胞的预测值
                #
                # if max < temp:
                #     max = temp
                # if temp > 0.1:
                    # print('准确率大于0.5的匹配情况:', temp, "左图细胞序号:", num1[k], "右图细胞序号：", num2[j])
                score_list.append(temp)  # 保存准确率
                # distance_list.append(distance)
                # print("列表1的维度:", np.array(score_list).shape)

                dict = '%s' % temp
                score[dict] = [k, j, num1[k], num2[j]]  # j,k, j+1, num[k] j k 程序中的序号， j + 1, num[k] 实际的文件名 >=1
            #  if temp>0.80:
            #      cv2.imwrite("Debug/%.6d_1.jpg" % k, image1[j] * 255)
            #      cv2.imwrite("Debug/%.6d_2.jpg" % k, image2[k] * 255)
        score_list1 = np.array(score_list).reshape(b, a) #网络输出的相似值矩阵
        # distance_list=np.array(distance_list).reshape(b, a) #手动设计距离矩阵
        # score_list2=np.multiply(score_list1,distance_list)

        print("图:",name[:-4],"列表2的维度:{}", np.array(score_list1).shape)

        matcher = KMMatcher(score_list1)
        best, weight = matcher.solve(verbose=True)#best：字典 保存下一帧序号 上一帧序号 权重

        weight = np.array(weight) #对匹配度降序排列
        weight = -weight
        weight_sort = weight.argsort()

        s = 1
        for k in range(0, b):  # crop_result/..._2的每个细胞块
            match_data = best.get(str(weight_sort[k]))
            max_score = best.get(str(weight_sort[k]))  # 匹配值最大的对应细胞的序号和4通道图像，两个序号，第一张图和第二张图  max_score[0] 上一帧大图序号
            if max_score[2] > 0.5: #高于这个阈值的才认为是匹配的
                print("大图{0},find:{1},上一张{2}号和下一张{3}号细胞权重：{4}".format(name[:-4], s, int(max_score[1]), int(max_score[0]),max_score[2]))
                src1_coordinate = txt1[int(max_score[1]) - 1]  # 上一张图对应的细胞序号的中心细胞的坐标
                x1 = float(src1_coordinate[0])  # 横坐标
                y1 = float(src1_coordinate[1])  # 纵坐标
                # print(src1_coordinate)
                # 上一帧原图编号
                # j = int(max_score[0] - 1)  # 上一帧的序号 j
                # k = int(max_score[1] - 1)  # 下一帧匹配的序号 num[k]-1
                cell_mark_num = s  # 给一对匹配大图标号 j位置的数字
                src1 = cv2.putText(src1, '%d' % cell_mark_num, (int(x1 - 5), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                   (0, 255, 0), 1)  # 标号
                cv2.imwrite('result/' + name[:-4] + '_1.jpg', src1)  # 在原图上标号
                src2_coordinate = txt2[int(max_score[0]) - 1]  # 第二张图对应的细胞1坐标
                x2 = float(src2_coordinate[0])
                y2 = float(src2_coordinate[1])
                src2 = cv2.putText(src2, '%d' % cell_mark_num, (int(x2 - 5), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                   (0, 255, 0), 1)
                cv2.imwrite('result/' + name[:-4] + '_2.jpg', src2)
                s = s + 1

                del num3[num3.index(int(max_score[1]))]  # 删除上一帧匹配上的
                del num4[num4.index(int(max_score[0]))]  # 删除下一帧匹配上的


