import json
import time
from pathlib import Path
import numpy as np
import torch
import os
import cv2
import pandas as pd

class SemanticSegmentationModel:

    def __init__(self):
        # Global variables
        self.root_dir = "./../src/customDataset/"
        self.label_colors_file = os.path.join(self.root_dir, "label_colors.json")
        self.label2color = {}
        self.color2label = {}
        self.label2index = {}
        self.index2label = {}
        # Model globals
        self.n_class = 5
        self.means = np.array([103.939, 116.779, 123.68]) / 255.

        #Load model
        model_path = "../src/models/trainRound2/FCNs-BCEWithLogits_batch2_epoch250_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05"
        self.model = torch.load(model_path)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.model = self.model.cuda()
        self.model.eval()

        #Parse the labels
        self.parse_labels()


    def parse_labels(self,):
        f = json.load(open(self.label_colors_file, "r"))

        for idx, cl in enumerate(f):
            label = cl["name"]
            color = cl["color"]
            print(label, color)

            self.label2color[label] = color
            self.color2label[tuple(color)] = label
            self.label2index[label] = idx
            self.index2label[idx] = label

    def calculate_segmentation(self, img):
        h, w, _ = img.shape
        assert w == 640 and h == 480, 'Image size should be 640x480'

        val_h = h; val_w = w

        orig_img = np.copy(img)

        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        inputs = torch.from_numpy(img.copy()).float()
        inputs = torch.unsqueeze(inputs, 0).cuda()
        output = self.model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        assert (N == 1)
        pred = output.transpose(0, 2, 3, 1).reshape(-1, self.n_class).argmax(axis=1).reshape(h, w)

        pred_img = np.zeros((val_h, val_w, 3), dtype=np.float32)
        for cls in range(self.n_class):
            pred_inds = pred == cls
            label = self.index2label[cls]
            color = self.label2color[label]
            pred_img[pred_inds] = color

        pred_img = pred_img.astype(np.uint8)

        pred_img = cv2.cvtColor(pred_img,cv2.COLOR_RGB2BGR)
        return pred_img


if __name__ == "__main__":
    debug = False
    bloodCode  = 230+20+20
    videoPath = r'C:\Users\asus\OneDrive - purdue.edu\ThesisDataset\03-PauCollection\2021-03-12_12h.57m.43s_pau_automy01'
    videoPath = Path(videoPath) / '00_video_right_color.avi'
    cap = cv2.VideoCapture(str(videoPath)) #Open cv not compatible with pathlib objects

    # Check if camera opened successfully
    if not cap.isOpened():
      print("Error opening video stream or file")

    #Output video
    bloodCalculationPath = videoPath.parent / "blood_percentage.txt"
    bloodCalculation = pd.DataFrame(columns= ["blood_percentage"])
    bloodCalculation.index.name = "frame_id"
    resultPath = videoPath.parent / (videoPath.name + "_processed.avi")

    frame_width = 640
    frame_height = 480
    totalNumbOfPix = 640*480
    out = cv2.VideoWriter(str(resultPath), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    #Create deep learning model for segmentation
    fcnModel = SemanticSegmentationModel()
    init_time = time.time()
    count = 0
    while cap.isOpened():
      ret, frame = cap.read() # Capture frame-by-frame
      if ret:

        segm = fcnModel.calculate_segmentation(frame)
        out.write(segm)

        # Calculate percentage of bleeding
        numbOfPix = np.sum(segm, axis=2)
        bloodPix = np.sum(numbOfPix == bloodCode)
        bloodPix = bloodPix / totalNumbOfPix
        bloodCalculation.loc[count] = [bloodPix]
        count+=1
        frame_rate = (time.time() - init_time)*1000
        init_time = time.time()
        print("frame {:06d}, rate {:0.3f}ms".format(count,frame_rate))
        if debug:
            cv2.imshow('Frame',segm)
            cv2.imshow('Original',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      else:
        break

    # When everything done, release the video capture object
    bloodCalculation.to_csv(bloodCalculationPath)
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

