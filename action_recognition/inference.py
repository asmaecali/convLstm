import os
import cv2
import random
import numpy as np
import tensorflow as tf
from collections import deque

# Définir les constantes
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 48
CLASSES_LIST = ["normal", "anormal"]


DATASET_DIR = "/home/c3po/AR/Convlstm/data_client0/normal/0ae48cbe-634e-41ba-ac23-12c2aef47fad_82914_1920_alpha_00.mp4"
MODEL_PATH = "/home/c3po/AR/Convlstm/LRCN_model___Date_Time_2024_07_30__11_56_56___Loss_0.5951205492019653___Recall_0.7733333110809326.h5"

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

def predict_single_action(video_file_path, model):
    frames_list = frames_extraction(video_file_path)
    
    if len(frames_list) == SEQUENCE_LENGTH:
        predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label]

        print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]:.4f}')
    else:
        print("Not enough frames extracted to make a prediction")

"""def process_videos_in_directory(directory_path, model_file_path):
    # Charger le modèle une seule fois
    model = tf.keras.models.load_model(model_file_path)
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                video_file_path = os.path.join(root, file)
                print(f"\nProcessing video: {video_file_path}")
                predict_single_action(video_file_path, model)

# Traiter les vidéos dans le dossier spécifié"""
predict_single_action(DATASET_DIR, MODEL_PATH)
