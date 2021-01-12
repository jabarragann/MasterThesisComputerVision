from matplotlib import pyplot as plt
import numpy as np
import random
import os
import cv2
from PIL import Image
import json


def divide_train_val(val_rate=0.1, shuffle=True, random_seed=None):
    data_list = os.listdir(data_dir) #Folder that includes the images
    data_len = len(data_list)
    val_len = int(data_len * val_rate)

    if random_seed:
        random.seed(random_seed)

    if shuffle:
        data_idx = random.sample(range(data_len), data_len)
    else:
        data_idx = list(range(data_len))

    val_idx = [data_list[i] for i in data_idx[:val_len]]
    train_idx = [data_list[i] for i in data_idx[val_len:]]

    # create val.csv
    v = open(val_label_file, "w")
    v.write("img,label\n")
    for idx, name in enumerate(val_idx):
        if 'png' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
        lab_name = lab_name + "___fuse.png.npy"
        v.write("{},{}\n".format(img_name, lab_name))

    # create train.csv
    t = open(train_label_file, "w")
    t.write("img,label\n")
    for idx, name in enumerate(train_idx):
        if 'png' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
        lab_name = lab_name + "___fuse.png.npy"
        t.write("{},{}\n".format(img_name, lab_name))


def parse_label():
    # change label to class index
    f = json.load(open(label_colors_file, "r"))

    for idx, cl in enumerate(f):
        label = cl["name"]
        color = cl["color"]
        print(label, color)
        label2color[label] = color
        color2label[tuple(color)] = label
        label2index[label] = idx
        index2label[idx] = label

    for idx, name in enumerate(os.listdir(label_dir)):
        filename = os.path.join(label_idx_dir, name)

        #If the image has already been processed, skip it.
        if os.path.exists(filename + '.npy'):
            print("{:} Skip {:}".format(idx, name))
            continue

        print("Parse %s" % (name))

        #Load image
        img_name = os.path.join(label_dir, name)
        img = Image.open(img_name)
        img = np.array(img)

        height, weight, _ = img.shape

        idx_mat = np.zeros((height, weight))
        for h in range(height):
            for w in range(weight):
                #The images have 4 color channels (RGB and transparency)
                color = tuple(img[h, w][:-1])
                try:
                    label = color2label[color]
                    index = label2index[label]
                    idx_mat[h, w] = index
                except:
                    print("error: img:%s, h:%d, w:%d, {%s}" % (name, h, w, str(color)))
        idx_mat = idx_mat.astype(np.uint8)
        np.save(filename, idx_mat)
        print("Finish %s" % (name))

    # test some pixels' label
    # img = os.path.join(label_dir, os.listdir(label_dir)[0])
    # # img = scipy.misc.imread(img, mode='RGB')
    # img = Image.open(img)
    # img = np.array(img)
    # test_cases = [(127, 110), (68,214), (238,225), (5,5)]
    # test_ans = ['Blood', 'Arms', 'Vein', 'Background']
    # for idx, t in enumerate(test_cases):
    #     color = img[t]
    #     assert color2label[tuple(color[:-1])] == test_ans[idx]


#############################
# global variables #
#############################
root_dir = "customDataset/"
data_dir = os.path.join(root_dir, "img")  # train data
label_dir = os.path.join(root_dir, "labels")  # train label
label_colors_file = os.path.join(root_dir, "label_colors.json")  # color to label
val_label_file = os.path.join(root_dir, "val.csv")  # validation file
train_label_file = os.path.join(root_dir, "train.csv")  # train file

# create dir for label index
label_idx_dir = os.path.join(root_dir, "labelsProcessed")
if not os.path.exists(label_idx_dir):
    os.makedirs(label_idx_dir)

label2color = {}
color2label = {}
label2index = {}
index2label = {}

if __name__ == '__main__':
    divide_train_val(random_seed=1)
    parse_label()
