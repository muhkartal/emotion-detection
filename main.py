import cv2
import pygame
from pathlib import Path
import time

BODY_PARTS = {
    "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
    "Background": 15
}

POSE_PAIRS = [
    ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]
]

BASE_DIR = Path(__file__).resolve().parent
protoFile = "C:/Users/iKartal/Kod/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "C:/Users/iKartal/Kod/pose_iter_160000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

capture = cv2.VideoCapture(0)

inputWidth = 320
inputHeight = 240
inputScale = 1.0 / 255

blanksound_path = str(BASE_DIR) + "/blanksound.mp3"
happy_music_path = str(BASE_DIR) + "/happy_bgm.mp3"
sad_music_path = str(BASE_DIR) + "/sad_bgm.mp3"

pygame.mixer.init()
pygame.mixer.music.stop()
happy_pose = False
crying_pose = False
start_time = time.time()
elapsed_time = 0

while cv2.waitKey(1) < 0:
    hasFrame, frame = capture.read()

    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    points = []
    for i in range(0, 15):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]
        if prob > 0.1:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)

    neck = points[BODY_PARTS["Neck"]]
    chest = points[BODY_PARTS["Chest"]]
    right_shoulder = points[BODY_PARTS["RShoulder"]]
    left_shoulder = points[BODY_PARTS["LShoulder"]]
    right_elbow = points[BODY_PARTS["RElbow"]]
    left_elbow = points[BODY_PARTS["LElbow"]]
    right_wrist = points[BODY_PARTS["RWrist"]]
    left_wrist = points[BODY_PARTS["LWrist"]]
    right_hip = points[BODY_PARTS["RHip"]]
    left_hip = points[BODY_PARTS["LHip"]]
    right_knee = points[BODY_PARTS["RKnee"]]
    left_knee = points[BODY_PARTS["LKnee"]]
    elbow_to_hip = None

    if right_shoulder and right_wrist and left_shoulder and left_wrist and chest and right_knee:
        shoulder_to_wrist = abs(right_shoulder[0] - right_wrist[0]) + abs(left_shoulder[0] - left_wrist[0])
        chest_to_knee = abs(chest[1] - right_knee[1])
        if shoulder_to_wrist < 100 and chest_to_knee < 100:
            crying_pose = True
        else:
            crying_pose = False

    if right_elbow and left_elbow and right_hip and left_hip and left_knee and neck:
        elbow_to_hip = abs(right_elbow[1] - right_hip[1]) + abs(left_elbow[1] - left_hip[1])
        neck_to_knee = abs(neck[1] - left_knee[1])
        if elbow_to_hip > 200 and neck_to_knee > 50:
            happy_pose = True
        else:
            happy_pose = False

    elapsed_time = time.time() - start_time
    if elapsed_time >= 10:
        print("elbow_to_hip:", elbow_to_hip)
        start_time = time.time()

    if crying_pose:
        if pygame.mixer.music.get_busy() == 0 or pygame.mixer.music.get_pos() == -1:
            pygame.mixer.music.load(sad_music_path)
            pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.stop()

    if happy_pose:
        if pygame.mixer.music.get_busy() == 0 or pygame.mixer.music.get_pos() == -1:
            pygame.mixer.music.load(happy_music_path)
            pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.stop()

    for pair in POSE_PAIRS:
        partA = BODY_PARTS[pair[0]]
        partB = BODY_PARTS[pair[1]]
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

    cv2.imshow("Output-Keypoints", frame)

capture.release()
cv2.destroyAllWindows()
