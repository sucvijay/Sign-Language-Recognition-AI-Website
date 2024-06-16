# import pickle

# import cv2
# import mediapipe as mp
# import numpy as np

# import torch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.conv1 = nn.Conv2d(1, 80, kernel_size = 5)
#         self.conv2 = nn.Conv2d(80, 80, kernel_size = 5)

#         self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
#         self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

#         self.batch_norm1 = nn.BatchNorm2d(80)
#         self.batch_norm2 = nn.BatchNorm2d(80)

#         self.fc1 = nn.Linear(1280, 250)
#         self.fc2 = nn.Linear(250, 25)

#     def forward(self, x):

#         x = self.conv1(x)
#         x = self.batch_norm1(x)
#         x = F.relu(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.batch_norm2(x)
#         x = F.relu(x)
#         x = self.pool2(x)

#         x = x.view(x.size(0), -1)

#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = F.log_softmax(x, dim=1)

#         return x

# cap = cv2.VideoCapture(1)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # labels_dict = {0: 'Yes', 1: 'No', 2: 'Thank You',3:"Hello", 4:"I Love You",5:"Good Bye", 6:"You are Welcome",7:"Please",8:"sorry"}

# signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
#         '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
#         '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }


# modelo = torch.load('model_trained.pt')
# # modelo.eval()

# while True:

#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * (W)) - 10
#         y1 = int(min(y_) * (H)) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
        
#         # 
#         tol = 25
#         img = frame[y1-tol:y2+tol, x1-tol:x2+tol]
#         cv2.imshow("cropped",img)
        
#         res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
#         res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

#         res1 = np.reshape(res, (1, 1, 28, 28)) / 255
#         res1 = torch.from_numpy(res1)
#         res1 = res1.type(torch.FloatTensor)
        
#         out = modelo(res1)
#         # Probabilidades
#         probs, label = torch.topk(out, 25)
#         probs = torch.nn.functional.softmax(probs, 1)

#         pred = out.max(1, keepdim=True)[1]

#         if float(probs[0,0]) < 0.5:
#             texto_mostrar = 'Sign not detected'
#         else:
#             texto_mostrar = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0])) + '%'
#             cv2.putText(frame, texto_mostrar , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#             cv2.LINE_AA)
        
        
        
        


#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)


# cap.release()
# cv2.destroyAllWindows()
