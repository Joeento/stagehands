import argparse
import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

SOURCE_DIR ='source/'
DEST_DIR = 'dest/'

def match(filename):
    img = cv2.imread(SOURCE_DIR + filename)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cv2.imwrite(DEST_DIR + filename, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate pose for a given image')
    parser.add_argument('filename', action='store', default='dance.jpg', help='name of file in source folder to process')

    args = parser.parse_args()
    filename = args.filename

    match(filename)
