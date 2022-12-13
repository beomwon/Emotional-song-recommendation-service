from torch import softmax
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
model = torch.load("pt\\electra.pt",map_location=device)

def emotion_encodeing_text(text):
    encodeing = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True, add_special_tokens=True)
    return encodeing

def emotion_predict(encodeing):
    global model
    
    input_ids = encodeing['input_ids'].to(device)
    attention_mask = encodeing['attention_mask'].to(device)

    with torch.no_grad():
        model = model.eval()
        predict = model(input_ids, attention_mask)[0].to(device)
        predict = softmax(predict, dim=1).cpu()
        predict = predict.detach().numpy()[:,:].tolist()[0]

    return predict

def predict(sentence):
    encode_text = emotion_encodeing_text(sentence)
    predicts = emotion_predict(encode_text)
    prediction=[]

    for i in predicts:
        i = round(i,3)
        prediction.append(round(float(i)*100, 3))
        
    return prediction

if __name__ == '__main__':
    predict("화남")