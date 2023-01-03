import argparse
import cv2
import mediapipe as mp
import time
import math

from numpy import dot
from numpy.linalg import norm

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

SOURCE_DIR ='source/'
DEST_DIR = 'dest/'

def cosine_distance_matching(A, B):
    cos_sim = dot(a, b)/(norm(a) * norm(b))
    return cos_sim

    distance = 2 * (1 - cos_sim);
    return math.sqrt(distance);

def read_poses(image1, image2):
    imgRGB1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    pose1 = pose.process(imgRGB1)

    imgRGB2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    pose2 = pose.process(imgRGB2)

    return pose1, pose2

def match(filename1, filename2):
    image1 = cv2.imread(SOURCE_DIR + filename1)
    image2 = cv2.imread(SOURCE_DIR + filename2)
    pose1, pose2 = read_poses(image1, image2)

    if pose1.pose_landmarks:
        mpDraw.draw_landmarks(image1, pose1.pose_landmarks, mpPose.POSE_CONNECTIONS)
        print("Pose 1")
        print(pose1.pose_landmarks)

    pose_vector_1 = [landmark.x, landmark.y for landmark in pose1.pose_landmarks]
    pose_vector_2 = [landmark.x, landmark.y for landmark in pose2.pose_landmarks]
    print("CDM: " + cosine_distance_matching(pose_vector_1, pose_vector_2))

    cv2.imwrite(DEST_DIR + filename1, image1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate pose for a given image')
    parser.add_argument('filename1', action='store', default='dance.jpg', help='name of original file with pose data')
    parser.add_argument('filename2', action='store', default='dance.jpg', help='name of the file we want to compare to original')

    args = parser.parse_args()
    filename1 = args.filename1
    filename2 = args.filename1

    match(filename1, filename2)
