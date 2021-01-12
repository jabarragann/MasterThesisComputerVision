import json

import numpy as np
import scipy.misc
import torch
import os
import cv2
from PIL import Image

def parse_label():
    f = json.load(open(label_colors_file, "r"))

    for idx, cl in enumerate(f):
        label = cl["name"]
        color = cl["color"]
        print(label, color)
        label2color[label] = color
        color2label[tuple(color)] = label
        label2index[label] = idx
        index2label[idx] = label


def test_img(img_path):
    # img = scipy.misc.imread(img_path, mode='RGB')
    img = Image.open(img_path)

    w,h = img.size
    assert w == 640 and h == 480, 'Error with input shape'

    # h, w, c = img.shape[0], img.shape[1], img.shape[2]
    val_h = h
    val_w = w
    #resize image
    # img = scipy.misc.imresize(img, (val_h, val_w), interp='bilinear', mode=None)
    # img = cv2.resize(img, (val_h, val_w), interpolation=cv2.INTER_AREA)
    img = img.resize((val_w,val_h)) #PILLOW images resize methods requires a tuple with the (width,height). Different from cv2.resize


    img = np.array(img) ##Convert PIL imaged to numpy a array
    img = img[:, :, ::-1]  ## # switch to BGR
    orig_img = np.copy(img)

    #Debug
    # cv2.imshow('test',img)
    # cv2.waitKey(0)

    img = np.transpose(img, (2, 0, 1)) / 255.
    img[0] -= means[0]
    img[1] -= means[1]
    img[2] -= means[2]

    inputs = torch.from_numpy(img.copy()).float()
    inputs = torch.unsqueeze(inputs, 0).cuda()
    output = model(inputs)
    output = output.data.cpu().numpy()

    N, _, h, w = output.shape
    assert (N == 1)
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(h, w)


    pred_img = np.zeros((val_h, val_w, 3), dtype=np.float32)
    for cls in range(n_class):
        pred_inds = pred == cls
        label = index2label[cls]
        color = label2color[label]
        pred_img[pred_inds] = color

    cv2.namedWindow('Frame prediction', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask',cv2.WINDOW_NORMAL)

    pred_img = pred_img.astype(np.uint8)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)  #Change to BGR color space

    # finalFrame = np.hstack((orig_img, pred_img))
    finalFrame = 0.8*orig_img + 0.2*pred_img
    finalFrame = finalFrame.astype(np.uint8)

    #Show image
    cv2.imshow("mask", pred_img)
    cv2.imshow("Frame prediction", finalFrame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # pred_img = scipy.misc.imresize(pred_img, (h, w), interp='bilinear', mode=None)
    # scipy.misc.imsave('result.png', pred_img)

if __name__ == "__main__":
    #Global variables
    root_dir = "customDataset/"
    label_colors_file = os.path.join(root_dir, "label_colors.json")
    label2color = {}
    color2label = {}
    label2index = {}
    index2label = {}

    #Model globals
    n_class = 5
    means = np.array([103.939, 116.779, 123.68]) / 255.

    # model_path = "models/old/modelv1/FCNs-BCEWithLogits_batch2_epoch10_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05"
    model_path = "models/FCNs-BCEWithLogits_batch2_epoch150_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05"
    model = torch.load(model_path)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    model.eval()

    parse_label()
    # img_path = r"customDataset/img\white_vein_left_frames_scene12241.png"
    img_path = r"customDataset/img\red_vein_1_left_frames_scene14641.png"
    test_img(img_path)