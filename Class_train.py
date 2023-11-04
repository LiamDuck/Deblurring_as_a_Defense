import os
import random
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from models import *
from Train.train import train_classifier
from Train.train import test_classifier
from Train.train import classifier_acr
def main():
    train = datasets.SVHN("./Datasets", 
                        split="train",
                        transform=transforms.Compose([transforms.ToTensor()]),
                        download=True)
    test = datasets.SVHN("./Datasets",
                        split="test",
                        transform=transforms.Compose([transforms.ToTensor()]),
                        download=True)
    train = DataLoader(train, 128, True, num_workers=11)
    test_loader = DataLoader(test, 128, True, num_workers=11)
    test_eval = DataLoader(test, 128, True, num_workers=11)
    
    test_eval_list = []
    for data, label in test_eval:
        test_eval_list.append((data, label))
    
    device = torch.device("cuda")
    model = StreetNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    losses = [[], []]
    for i in range(epochs):
        print(f'Epoch #{i}')
        model.train(True)
        train_loss = train_classifier(model, device, train, opt)
        losses[0].append(train_loss)
        model.eval()
        test_loss = test_classifier(model, device, test_loader)
        losses[1].append(test_loss)

        print(f'train: {train_loss}')
        print(f'test: {test_loss}')
        if i % 10 == 9:
            makeLineGraph(losses)
            plt.close('all')
            acc = classifier_acr(model, device, test_eval)
            print('The model is: {:.2f}% accurate'.format(acc))
    torch.save(model.state_dict(), './Saves/svhn_class.pt')
    if(torch.cuda.is_available()):
        torch.cuda.empty_cache()
        
def makeLineGraph(losses):
    work_path = os.getcwd()
    fig = plt.figure()
    plt.plot(losses[0], label='training')
    plt.plot(losses[1], label='testing')
    plt.yscale('log')
    plt.legend(title='losses') 
    plt.savefig('./Outputs/svhn_class_graphs')
    
if __name__ == '__main__':
    main()