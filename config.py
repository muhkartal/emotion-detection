# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
protoFile = "C:/Users/iKartal/Kod/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "C:/Users/iKartal/Kod/pose_iter_160000.caffemodel"
inputWidth = 320
inputHeight = 240
inputScale = 1.0 / 255

blanksound_path = str(BASE_DIR) + "/blanksound.mp3"
happy_music_path = str(BASE_DIR) + "/happy_bgm.mp3"
sad_music_path = str(BASE_DIR) + "/sad_bgm.mp3"
