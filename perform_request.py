#lib
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torchvision
from torchvision import datasets, models, transforms
from datetime import datetime
import numpy as np
import time
import os
import random
from PIL import Image
import threading
import stylegan2

#py
import similar_celebrity
import face_emotion
import text_emotion
import songinfo

users, t = [], None

def save_image(f):
    now = datetime.now()
    user_image_name = str(now.microsecond)
    f.save('static\\uploads\\'+ user_image_name +'.jpg')
    return 'static\\uploads\\'+ user_image_name +'.jpg'

def emotion_result(face, text):
    return [face[0] *0.2 + text[1]*0.8, face[3]*0.2 + text[0]*0.8, face[2]*0.2 + text[2]*0.8, face[5]*0.2 + text[3]*0.8, face[4]*0.2 + text[4]*0.8, face[1]*0.2 + text[5]*0.8]

def select_photo(index, celebrity_list, gender):
    for root, dir, files in os.walk('static\\celebrityPhoto\\' + gender + '\\' + celebrity_list[index]):
        photos, albums = [], []
        for v in random.sample(files, 3):
            photos.append('static/celebrityPhoto/' + gender + '/' + celebrity_list[index] + '/' + v)
        for v in random.sample(files, 4):
            albums.append('static/celebrityPhoto/' + gender + '/' + celebrity_list[index] + '/' + v)
    return photos, albums

def init(image, comment, gender):
    e = ['기쁨', '불안', '분노', '슬픔', '상처', '당황']
    f = open('pt\\celebrity_list.pkl','rb')
    celebrity_list = pickle.load(f)
    f.close()

    user_image_root = save_image(image)
    face_emotion_result = face_emotion.predict(user_image_root) # 얼굴 감정 분석 결과 /['happy', 'anxious', 'angry', 'sad', 'hurt', 'surprised']
    text_emotion_result = text_emotion.predict(comment) #텍스트 분석 결과/ 불안, 기쁨, 분노, 슬픔, 상처, 당황 리스트 퍼센트로 나옴
    emotion = emotion_result(face_emotion_result, text_emotion_result) # max값이 emotion 최우선값임
    celebrity_results = similar_celebrity.predict(user_image_root, gender)
    
    if celebrity_results['face_preds']['celebrity_index'] == 0: celebrity = '10CM' #10cm로 저장되어 있어서 따로 빼놓음
    else: celebrity = celebrity_list[celebrity_results['face_preds']['celebrity_index']].split()[0]

    song_results = songinfo.process(celebrity, e[emotion.index(max(emotion))])
    total_emotion = ['happy', 'anxious', 'angry', 'sad', 'hurt', 'surprised'][emotion.index(max(emotion))]

    photos, albums = select_photo(celebrity_results['face_preds']['celebrity_index'], celebrity_list, gender)

    result = {'user_image': user_image_root.split('\\')[-1],'celebrity_name': celebrity_list[celebrity_results['face_preds']['celebrity_index']], 'celebrity_percent': round(celebrity_results['face_preds']['percent']*100, 3), 
              'eye': round(celebrity_results['eye_preds']['percent']*100, 3), 'nose': round(celebrity_results['nose_preds']['percent']*100, 3), 'lip': round(celebrity_results['lip_preds']['percent']*100, 3), 
              'face_emotion': face_emotion_result, 'text_emotion': text_emotion_result, 'song_result': song_results, 'total_emotion': total_emotion, 'photos': photos, 'albums': albums,
              'celebrity_category': celebrity_results['face_preds']['celebrity_index']}

    return result

def thread_gan(celebrity_cat, email, image, celebrity, total_emotion, song_result):
    global users, count, t
    users.append([celebrity_cat, email, image, celebrity, total_emotion, song_result])

    if t == None or not(t.is_alive()):
        # print('thread 진입')
        t = threading.Thread(target=stylegan2.photofunia, args=(users,))
        users = []
        t.start()
        
if __name__ == '__main__':

    f = open('pt\\celebrity_list.pkl','rb')
    celebrity_list = pickle.load(f)
    f.close()

    select_photo(0, celebrity_list, 'male')

