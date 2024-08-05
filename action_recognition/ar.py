import os
import cv2
import random
import numpy as np
import tensorflow as tf
from collections import deque

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 2
CLASSES_LIST = ["normal", "anormal"]
DATASET_DIR = "/home/c3po/AR/Convlstm/data_client2/test/anormal/1018_1710839628.mp4"
MODEL_PATH = "/home/c3po/AR/Convlstm/convlstm_model___2024_07_30__11_56_37_recall_0.6800000071525574.h5"

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list

def predict_single_action(video_file_path, model_file_path):
    convlstm_model = tf.keras.models.load_model(model_file_path)
    
    frames_list = frames_extraction(video_file_path)
    
    if len(frames_list) == SEQUENCE_LENGTH:
        frames_array = np.expand_dims(frames_list, axis=0)
        predicted_labels_probabilities = convlstm_model.predict(frames_array)[0]
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label]
        return predicted_class_name
    else:
        print("Not enough frames extracted to make a prediction")
        return None

def ar(video_path):
    model_path = "/home/c3po/Bureau/action_recognition/models/LRCN_model___Date_Time_2024_07_29__15_52_18___Loss_0.5458952188491821___Recall_0.7435897588729858.h5"
    predicted_class_name = predict_single_action(video_path, model_path)
    
    if predicted_class_name == "anormal":
        return False
    elif predicted_class_name == "normal":
        return True
    else:
        return None  


video_path = "data_client2/test/anormal/6c7e139c-1eab-4225-8e72-c5706f87cc59_189961_1920_alpha_00.mp4"
res = ar(video_path)
print(res)
