# -*-coding: UTF-8 -*-
import numpy as np
from genplate_advanced import *
import os
import pandas as pd
import pickle
import cv2
import os

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}

chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z"]


def rand_range(lo, hi):
    return lo + r(hi - lo)


def r(val):
    return int(np.random.random() * val)


def gen_rand():
    name = ""
    label = []
    label.append(rand_range(0, 31))
    label.append(rand_range(41, 65))
    for i in range(5):
        label.append(rand_range(31, 65))

    name += chars[label[0]]
    name += chars[label[1]]
    for i in range(5):
        name += chars[label[i + 2]]
    return name, label


def gen_sample(genplate_advanced, width, height):
    name, label = gen_rand()
    img = genplate_advanced.generate(name)
    img = cv2.resize(img, (width, height))
    # img = np.multiply(img, 1 / 255.0)
    # img = img.transpose(2, 0, 1)
    return label, name, img


def genBatch(batchSize, outputPath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    label_store = []
    for i in range(batchSize):
        print('create num:' + str(i))
        label, name, img = gen_sample(genplate_advanced, 120, 30)
        label_store.append(label)
        filename = os.path.join(outputPath, str(i).zfill(4) + ".jpg")
        # filename = os.path.join(outputPath, label + ".jpg")
        # filename = outputPath + '/' + str(label) + ".jpg"
        # print(filename)
        cv2.imwrite(filename, img)
    # label_store = pd.DataFrame(label_store)
    np.savetxt('label.txt', label_store)
    # label_store.to_csv('label.txt')


batchSize = 5000
path = './data/train_data'
font_ch = './font/platech.ttf'
font_en = './font/platechar.ttf'
bg_dir = './NoPlates'
genplate_advanced = GenPlate(font_ch, font_en, bg_dir)
genBatch(batchSize=batchSize, outputPath=path)

# create train label
a = np.loadtxt('label.txt')
b = np.zeros([batchSize, 65])
for i in range(batchSize):
    for j in range(7):
        b[i, int(a[i, j])] = int(a[i, j])

# create image train data
img_data = np.zeros([batchSize, 30, 120, 3])
for i in range(batchSize):
    img_path = path + '/' + str(i).zfill(4) + ".jpg"
    img_temp = cv2.imread(img_path)
    img_temp = np.reshape(img_temp, (30, 120, 3))
    img_data[i, :, :, :] = img_temp


print(b)
output = open('train_data.pkl', 'wb')
pickle.dump(img_data, output)
# output = open('train_label.pkl', 'wb')
# pickle.dump(b, output)