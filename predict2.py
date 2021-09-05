import torch
import torch.nn as nn
from torchvision.models import resnet101
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
import torch.functional as F
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import sys
import json

def predict_image(image_path):
    #print("prediciton in progress")
    image = Image.open(image_path).convert('RGB')

    transformation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_tensor = transformation(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    if cuda:
        image_tensor.cuda()

    input = Variable(image_tensor)
    output = model(input)

    index = output.data.numpy().argmax()
    return index

def parameters():
    hyp_param = open('param_predict.txt','r')
    param = {}
    for line in hyp_param:
        l = line.strip('\n').split(':')

def class_mapping(index):
    with open("class_mapping.json") as cm:
        data = json.load(cm)
    if index == -1:
        return len(data)
    else:
        return data[str(index)]

def segregate():
    with open("class_mapping.json") as cm:
        data = json.load(cm)
    try:
        os.mkdir(seg_dir)
        #print("Directory " , seg_dir ,  " Created ") 
    except OSError:
        return
    for x in range (0,len(data)):
        dir_path=seg_dir+"/"+data[str(x)]
        try:
            os.mkdir(dir_path)
            #print("Directory " , dir_path ,  " Created ") 
        except OSError:
            return
            #print("Directory " , dir_path ,  " already created")


path_to_model = "./models/"+'trained.model'
checkpoint = torch.load(path_to_model)
seg_dir="/home/mai/Dev/BPClassifier/pi/seg/"

cuda = torch.cuda.is_available()

num_class = class_mapping(index=-1)
print (num_class)
model = resnet101(num_classes = 2)

if cuda:
    model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint, map_location = 'cpu')

model.eval()

if __name__ == "__main__":
    images = os.listdir("/home/mai/Dev/BPClassifier/pi/")
    for x in range (0,len(images)):
        if 'md' not in images[x]:
            imagepath = "/home/mai/Dev/BPClassifier/pi/"+images[x]
            img = Image.open(imagepath).convert('RGB')
            prediction = predict_image(imagepath)
            name = class_mapping(prediction)
            segregate()
            save_path = seg_dir+"/"+name+"/"+images[x]
            img.save(save_path)
            if name == "bonus":
                print(images[x])

