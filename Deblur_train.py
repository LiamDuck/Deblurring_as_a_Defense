import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from models import *
from torch.utils.data import DataLoader
from Train.train import train_deblur
from Train.train import test_deblur
import os
import random 

def main():
    train = datasets.OxfordIIITPet("./Datasets", 
                        "trainval",
                        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5, True)]),
                        download=False)
    test = datasets.OxfordIIITPet("./Datasets",
                        "test",
                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5, True)]),
                        download=False)
    train = DataLoader(train, 512, True, num_workers=8)
    test_loader = DataLoader(test, 512, True, num_workers=8)
    test_eval = DataLoader(test, 1, True, num_workers=1)

    blur = torchvision.transforms.GaussianBlur(
        kernel_size=9,
        sigma=(1.0, 2))
    train_data = []
    for data, label in train:
        blur_data = blur(data)
        train_data.append((data, blur_data, label))

    test_data = []
    for data, label in test_loader:
        blur_data = blur(data)
        test_data.append((data, blur_data, label))

    test_eval_data = []
    for data, label in test_eval:
        blur_data = blur(data)
        test_eval_data.append((data, blur_data, label))

    device = torch.device("cuda")
    model_1 = Deblur3Chan()

    opt = torch.optim.Adam([{'params': model_1.conv1.parameters(), 'lr': 0.001},
                            {'params': model_1.conv2.parameters(), 'lr': 0.001},
                            {'params': model_1.conv3.parameters(), 'lr': 0.0001}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            opt,
            mode='min',
            patience=5,
            factor=0.5,
            verbose=True
        )
    epochs = 500
    model_1.to(device)
    losses = [[], []]
    for i in range(epochs):
        print(f'Epoch #{i}')
        model_1.train(True)
        train_loss = train_deblur(model_1, device, train_data, opt)
        losses[0].append(train_loss)
        model_1.eval()
        test_loss = test_deblur(model_1, device, test_data)
        losses[1].append(test_loss)
        scheduler.step(test_loss)

        print(f'train: {train_loss}')
        print(f'test: {test_loss}')
        if i % 20 == 9:
            makeLineGraph(losses)

            original, blurred, lable = test_eval_data[random.randint(0, len(test_eval_data))]
            blurred = blurred.to(device)
            model_1.eval()
            output = model_1(blurred)
            images = [[original, blurred, output]]
            makeImageGraph(images)
            plt.close('all')
            torch.save(model_1.state_dict(), './Saves/celebA_deblur.pt')
    if(torch.cuda.is_available()):
        torch.cuda.empty_cache()

def makeImageGraph(imageList):
    row = len(imageList)
    col = len(imageList[0])

    fig = plt.figure()
    idx = 1
    lables = ["Original", "Blurred", "Unblurred"]
    for list in imageList:
        for i, image in enumerate(list):
            img = fig.add_subplot(row, col, idx)
            pil = transforms.ToPILImage()
            img.imshow(pil(image.cpu().detach().squeeze()))
            img.set_title(lables[i])
            plt.axis('off')
            idx += 1
    plt.savefig('./Outputs/celebA_images')

def makeLineGraph(losses):
    work_path = os.getcwd()
    fig = plt.figure()
    plt.plot(losses[0], label='training')
    plt.plot(losses[1], label='testing')
    plt.yscale('log')
    plt.legend(title='losses') 
    plt.savefig('./Outputs/celebA_graphs')

if __name__ == '__main__':
    main()
