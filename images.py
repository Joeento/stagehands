import argparse
import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

parser = argparse.ArgumentParser(description='Estimate pose for a given image')
parser.add_argument('filename', action='store', default='dance.jpg', help='name of file in source folder to process')

source_dir ='source/'
dest_dir = 'dest/'

args = parser.parse_args()
filename = args.filename

img = cv2.imread(source_dir + filename)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = pose.process(imgRGB)
# print(results.pose_landmarks)
    
if results.pose_landmarks:
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
for id, lm in enumerate(results.pose_landmarks.landmark):
    h, w,c = img.shape
    # print(id, lm)
    cx, cy = int(lm.x*w), int(lm.y*h)
    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

cv2.imwrite(dest_dir + filename, img)
