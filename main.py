import cv2
import time
from pose_detection import detect_pose, capture
from body_parts import BODY_PARTS, POSE_PAIRS
from music import play_music, stop_music
from config import happy_music_path, sad_music_path

happy_pose = False
crying_pose = False
start_time = time.time()

while cv2.waitKey(1) < 0:
    hasFrame, frame = capture.read()
    if not hasFrame:
        cv2.waitKey()
        break

    points = detect_pose(frame)

    # Check for crying pose
    if all([points[BODY_PARTS[part]] for part in ["Neck", "Chest", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist", "LWrist", "RKnee"]]):
        shoulder_to_wrist = abs(points[BODY_PARTS["RShoulder"]][0] - points[BODY_PARTS["RWrist"]][0]) + abs(points[BODY_PARTS["LShoulder"]][0] - points[BODY_PARTS["LWrist"]][0])
        chest_to_knee = abs(points[BODY_PARTS["Chest"]][1] - points[BODY_PARTS["RKnee"]][1])
        crying_pose = shoulder_to_wrist < 100 and chest_to_knee < 100
    else:
        crying_pose = False

    # Check for happy pose
    if all([points[BODY_PARTS[part]] for part in ["RElbow", "LElbow", "RHip", "LHip", "LKnee", "Neck"]]):
        elbow_to_hip = abs(points[BODY_PARTS["RElbow"]][1] - points[BODY_PARTS["RHip"]][1]) + abs(points[BODY_PARTS["LElbow"]][1] - points[BODY_PARTS["LHip"]][1])
        neck_to_knee = abs(points[BODY_PARTS["Neck"]][1] - points[BODY_PARTS["LKnee"]][1])
        happy_pose = elbow_to_hip > 200 and neck_to_knee > 50
    else:
        happy_pose = False

    # Check elapsed time for debug purposes
    elapsed_time = time.time() - start_time
    if elapsed_time >= 10:
        print("Elapsed time:", elapsed_time)
        start_time = time.time()

    # Play sad music if crying pose is detected
    if crying_pose:
        play_music(sad_music_path)
    else:
        stop_music()

    # Play happy music if happy pose is detected
    if happy_pose:
        play_music(happy_music_path)
    else:
        stop_music()

    # Draw the pose skeleton
    for pair in POSE_PAIRS:
        partA = BODY_PARTS[pair[0]]
        partB = BODY_PARTS[pair[1]]
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Output-Keypoints", frame)

# Release the resources
capture.release()
cv2.destroyAllWindows()
