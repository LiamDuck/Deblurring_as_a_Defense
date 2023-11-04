import torchattacks
from Util.pg_bar import printProgressBar
import torch.nn.functional as F
import torch

def train_deblur(model, device, data_set, opt):
    model.to(device)
    epoch_loss = 0

    for i, data in enumerate(data_set):
        org_img, blur_data, label = data
        blur_data, org_img = blur_data.to(device), org_img.to(device) 

        blur_data.requires_grad = True
        opt.zero_grad()

        output = model(blur_data)
        loss = F.mse_loss(output, org_img)
        epoch_loss += loss.item()

        loss.backward()
        opt.step()
        printProgressBar(i, len(data_set))
    print()
    return epoch_loss/len(data_set)

def test_deblur(model, device, data_set):
    model.to(device)
    epoch_loss = 0

    for i, data in enumerate(data_set):
        org_img, blur_data, label = data
        blur_data, org_img = blur_data.to(device), org_img.to(device) 

        blur_data.requires_grad = True

        output = model(blur_data)
        loss = F.mse_loss(output, org_img)
        epoch_loss += loss.item()

        loss.backward()
        printProgressBar(i, len(data_set))
    print()
    return epoch_loss/len(data_set)

def train_classifier(model, device, data_set, opt):
    model.to(device)
    epoch_loss = 0

    for i, data in enumerate(data_set):
        img, label = data
        img, label = img.to(device), label.to(device) 

        img.requires_grad = True
        opt.zero_grad()

        output = model(img)
        loss = F.cross_entropy(output, label)
        epoch_loss += loss.item()

        loss.backward()
        opt.step()
        printProgressBar(i, len(data_set))
    print()
    return epoch_loss/len(data_set)
        
def test_classifier(model, device, data_set):
    model.to(device)
    epoch_loss = 0

    for i, data in enumerate(data_set):
        img, label = data
        img, label = img.to(device), label.to(device) 

        output = model(img)
        loss = F.cross_entropy(output, label)
        epoch_loss += loss.item()
        printProgressBar(i, len(data_set))
    print()
    return epoch_loss/len(data_set)

def classifier_acr(model, device, data_set):
    model.to(device)
    correct = 0
    total = 0
    
    for i, data in enumerate(data_set):
        img, label = data
        img, label = img.to(device), label.to(device)
        
        output = model(img)
        init_pred = output.max(1, keepdim=True)[1]

        printProgressBar(i, len(data_set))
        for x in range(len(label)):
            total += 1
            if label[x].item() == init_pred[x].item():
                correct += 1
    print()
    return (correct/total) * 100

def defence_acr(class_model,unblur_model, device, blur_func=None, attack=None, data_set=None):
    class_model = class_model.to(device)
    unblur_model = unblur_model.to(device)
    correct = 0
    output_list = []

    for i, data in enumerate(data_set):
        img, label = data
        img, label = img.to(device), label.to(device)
        test_img = img
        if attack != None:
            adv_img = attack(img, label)
            test_img = adv_img
            if blur_func != None:
                blurred = blur_func(adv_img)
                test_img = unblur_model(blurred)
                output_list.append((img, adv_img, blurred, test_img))

            
        output = class_model(test_img)
        init_pred = output.max(1, keepdim=True)[1]

        printProgressBar(i, len(data_set))
        if init_pred.item() == label.item():
            correct += 1
    print()
    return ((correct/len(data_set)) * 100, output_list)