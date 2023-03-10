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
    cos_sim = dot(A, B)/(norm(A) * norm(B))

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

    if pose2.pose_landmarks:
        mpDraw.draw_landmarks(image2, pose2.pose_landmarks, mpPose.POSE_CONNECTIONS)
        print("Pose 2")
        print(pose2.pose_landmarks)

    pose_vector_1 = []
    for id, landmark in enumerate(pose1.pose_landmarks.landmark):
        print(id)
        pose_vector_1.append(landmark.x)
        pose_vector_1.append(landmark.y)

    pose_vector_2 = []
    for id, landmark in enumerate(pose2.pose_landmarks.landmark):
        pose_vector_2.append(landmark.x)
        pose_vector_2.append(landmark.y)
    print("CDM: " + str(cosine_distance_matching(pose_vector_1, pose_vector_2)))

    cv2.imwrite(DEST_DIR + filename1, image1)
    cv2.imwrite(DEST_DIR + filename2, image2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate pose for a given image')
    parser.add_argument('filename1', action='store', default='dance.jpg', help='name of original file with pose data')
    parser.add_argument('filename2', action='store', default='dance.jpg', help='name of the file we want to compare to original')

    args = parser.parse_args()
    filename1 = args.filename1
    filename2 = args.filename2

    match(filename1, filename2)
