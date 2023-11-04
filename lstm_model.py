import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.num_layers = 2
        self.lstm_size = 300

        self.lstm = nn.LSTM(input_size=66, hidden_size=self.lstm_size,
                            num_layers=self.num_layers, batch_first=True,dropout=0.2)
        self.fc1 = nn.Linear(in_features=self.lstm_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=797)

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.lstm_size).to('cpu'),
                torch.zeros(self.num_layers, batch_size, self.lstm_size).to('cpu'))

    def forward(self, x):
        x, s = self.lstm(x, self.init_state(x.size()[0]))
        x = x[:,-1,:]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x,s

model = LSTM()
model.load_state_dict(torch.load('D:/Praxis/capstone/hand_sign_model.pth', map_location=torch.device('cpu')))


class Data:
    def __init__(self) -> None:
        pass

    def video_data(video):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        cap = cv2.VideoCapture(video)
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        x = []
        y = []
        length = []

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if not results.pose_landmarks:
                    continue
                for idx,landmark in enumerate(results.pose_landmarks.landmark):
                    x.append(landmark.x)
                    y.append(landmark.y)
        cap.release()
        cv2.destroyAllWindows()
        length = x
        length.extend(y)
        length_arr = np.array(length).reshape(number_of_frames,66)
        return length_arr,number_of_frames
    
    def prepare_data(demo_input):
        X = []
        i = 0
        while True:
          seq_len=15
          # if (seq_len+i) >= math.ceil(len(demo_input)*0.7):
          if (seq_len+i) < len(demo_input):
            x,y = [],[]
            x.append(demo_input[i:seq_len+i])
            X.append(x)
            i+=1
          else:
            break
        return np.vstack(X)

