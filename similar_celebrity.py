#-*- coding: utf-8 -*-

import cv2
import torch
from PIL import Image
from torchvision import transforms
import dlib
import pickle
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

if device == 'cuda': torch.cuda.manual_seed_all(777)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('pt\\shape_predictor_68_face_landmarks.dat')

with open('pt\\celebrity_list.pkl','rb') as f:
    celebrity_list = pickle.load(f)

def find_point(shape, point_list, max_, min_):
    for f in point_list:
        part = shape.part(f)

        if part.y < min_: min_ = part.y
        if part.y > max_: max_ = part.y

    return min_, max_

def put_image_model(root, y, image):
    global device

    model = torch.load(root, map_location=torch.device('cuda')) #로드할 모델 넣기
    model.eval()
    
    transforms_image = transforms.Compose([transforms.Resize((y, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.fromarray(image)
    image = transforms_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, model_preds = torch.max(outputs, 1)

        temp = torch.round(torch.softmax(outputs, dim=1), decimals=3)
        result = []
        for v in temp[0]:
            result.append(round(float(v), 3))

    return {'celebrity_index': int(model_preds[0]), 'percent': max(result)}

def predict(root, gender): #img에 이미지 경로
    global celebrity_list

    eye_list = [x for x in range(17, 27)] + [x for x in range(36, 48)]
    nose_list = [x for x in range(27, 36)]
    lip_list = [x for x in range(48, 68)]

    src = cv2.imread(root)
    rects = detector(src[...,::-1], 1)

    for rect in rects: 
        x1, x2, y1, y2 = rect.left(), rect.right(), rect.top(), rect.bottom()
        shape = predictor(src, rect)

        face = src[y1: y2, x1: x2]
        min_eye, max_eye = find_point(shape, eye_list, y1, y2)
        min_nose, max_nose = find_point(shape, nose_list, y1, y2)
        min_lip, max_lip = find_point(shape, lip_list, y1, y2)
        
        eye  = src[min_eye: max_eye, x1: x2]
        nose = src[min_nose: max_nose, x1: x2]
        lip  = src[min_lip: max_lip, x1: x2]

    if gender == 'male':
        face_preds = put_image_model('pt\\face_model_man_70', 224, face)
        eye_preds = put_image_model('pt\\save model_eye_man_66.3', 75, eye)
        nose_preds = put_image_model('pt\\save model_nose 68', 75, nose)
        lip_preds = put_image_model('pt\\face_model_lip_60.8', 75, lip)

        return {'face_preds': face_preds, 'eye_preds': eye_preds, 'nose_preds': nose_preds, 'lip_preds': lip_preds}

    else:
        face_preds = put_image_model('pt\\face_model_woman_69', 224, face); face_preds['celebrity_index'] += 59
        eye_preds = put_image_model('pt\\model_eye_woman', 75, eye); eye_preds['celebrity_index'] += 59
        nose_preds = put_image_model('pt\\model_nose_woman_66', 75, nose); nose_preds['celebrity_index'] += 59
        lip_preds = put_image_model('pt\\model_lip_woman', 75, lip); lip_preds['celebrity_index'] += 59

        return {'face_preds': face_preds, 'eye_preds': eye_preds, 'nose_preds': nose_preds, 'lip_preds': lip_preds}
    

if __name__ == '__main__':
    print(predict('static\\uploads\\1224.jpg', 'male'))
