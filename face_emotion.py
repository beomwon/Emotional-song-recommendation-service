import torch
from torchvision import transforms
from PIL import Image

def predict(root):
    # ['happy', 'surprised', 'angry', 'anxious', 'hurt', 'sad']
    device = torch.device('cuda')
    model = torch.load('pt\\emotion_renet', map_location=torch.device('cuda'))
    transforms_test = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image = Image.open(root)
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        
        temp = torch.round(torch.softmax(outputs, dim=1), decimals=3)
        result = []
        for v in temp[0]:
            result.append(round(float(v)*100, 3))

        result = [result[6]+result[0], result[1], result[2], result[3], result[4], result[5]]
        return result
        # return round(float(temp[0][0]), 4)

    # return class_names[preds[0]]

if __name__ == '__main__':
    print(predict('27.jpg'))