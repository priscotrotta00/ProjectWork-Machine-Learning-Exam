import cv2, os, argparse, random
import numpy as np
from model import *

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    parser.add_argument("--weights_path", type=str, default='CNN+LSTM_best_model.pth', help="Weights")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method
WEIGHT_PATH = args.weights_path
WINDOW_SIZE = 4
MIN_DURATION = 2
THRESHOLD = 0.7

model = create_CNN_LSTM()
model = model.cuda() if torch.cuda.is_available() else model
transf = transformation()

if not torch.cuda.is_available():
  model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))
else:
  model.load_state_dict(torch.load(WEIGHT_PATH))

num_videos: int = len(os.listdir(args.videos))

num_frames: int = 0
frames = []

def get_window_predictions(video_frames, model, window_size):
    num_frames = video_frames.size(0)
    predictions = []

    for i in range(0, num_frames, 2):
        window_end = i+window_size if i+window_size <= num_frames else num_frames
        window = video_frames[i:i+window_size]
        window = window.unsqueeze(0)  # Aggiunge la dimensione del batch
        
        with torch.no_grad():
            output = model(window)
            predictions.append(output.item())

    return predictions

def classify_video(predictions, threshold=0.5, min_duration=10):
    video_length = len(predictions)
    start_frame = None
    fire_duration = 0

    for frame_idx in range(video_length):
        if predictions[frame_idx] >= threshold:
            if start_frame is None:
                start_frame = frame_idx
            fire_duration += 1
        else:
            if fire_duration >= min_duration:
                return 1, start_frame  # Video classificato come "presente di fuoco"

            start_frame = None
            fire_duration = 0

    if fire_duration >= min_duration:
        return 1, start_frame  # Video classificato come "presente di fuoco"
    else:
        return 0, None  # Video classificato come "non presente di fuoco"

################################################

# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(os.path.join(args.videos, video))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method
        if ret == True:
          if num_frames % (fps/2) == 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transf(image=img)['image']
            frames.append(img)
            img = None
          num_frames += 1
        else:
          frames_tensor = ImglistOrdictToTensor.forward(frames)
          frames_tensor = frames_tensor.cuda() if torch.cuda.is_available() else frames_tensor
          frames = []
        ########################################################
    cap.release()
    f = open(args.results+video+".txt", "w")

    # Here you should add your code for writing the results
    with torch.no_grad():
      model.eval()
      window_predictions = get_window_predictions(frames_tensor, model, WINDOW_SIZE)
      min_duration = MIN_DURATION if len(window_predictions) >= MIN_DURATION else len(window_predictions)
      prediction, start_window = classify_video(window_predictions, THRESHOLD, MIN_DURATION)
      if prediction:
        f.write(str(int(start_window)))

    ########################################################
    f.close()