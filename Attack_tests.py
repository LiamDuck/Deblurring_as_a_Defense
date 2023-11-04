import sys
import torch
import torchvision
import torchattacks
import threading
import time
from Util.pg_bar import printProgressBar
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from models import *
def run(eps, step, blurs, file):
    batch_size = 64
    test = datasets.SVHN("./Datasets",
                        'test',
                        transforms.Compose([transforms.ToTensor()]),
                        download=True)
    test_eval = DataLoader(test, batch_size, True, num_workers=12)

    model = StreetNetwork()
    model.load_state_dict(torch.load('./Saves/svhn_class.pt'))

    if torch.cuda.is_available():
        useDevice = 'cuda'
    else:
        useDevice = 'cpu'
    print(f'device: {useDevice}')
    device = torch.device(useDevice)

    deblur_model = Deblur3Chan()
    deblur_model.load_state_dict(torch.load('./Saves/svhn_deblur.pt'))
    deblur_model.to(device)
    deblur_model.eval()
    model = model.to(device)
    model.eval()

    pgd = torchattacks.PGD(model, eps, steps=step)
    pgd.device = device
    count = 0
    total = 0
    for i, set in enumerate(test_eval):
        data, label = set
        data = pgd(data, label)
        data, label = data.to(device), label.to(device)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        for x in range(len(label)):
            total += 1
            if label[x].item() == init_pred[x].item():
                count += 1
        printProgressBar(i, len(test_eval))
    print()
    log = f'no blurring: eps: {eps}, steps: {step}, accuracy: {(count/total * 100):.2f}% \n'
    file.write(log)
    print(log)


    for blur_pow in blurs:
        count = 0
        total = 0
        blur = torchvision.transforms.GaussianBlur(
            kernel_size=9,
            sigma=blur_pow)
        for i, set in enumerate(test_eval):
            data, label = set
            data = pgd(data, label)
            data = blur(data)
            data, label = data.to(device), label.to(device)
            data = deblur_model(data)
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]
            for x in range(len(label)):
                total += 1
                if label[x].item() == init_pred[x].item():
                    count += 1
            printProgressBar(i, len(test_eval))
        print()
        log = f'with blurring: eps: {eps}, steps: {step}, strength: {blur_pow}, accuracy: {(count/total * 100):.2f}% \n'
        file.write(log)
        print(log)

def main():
    file = open("./svhn_log.txt", "w")
    epses = [0.0, 0.05, 0.10, 0.15]
    steps = [50]
    blurs = [1.5, 1.75, 2.0]
    start = time.time()
    for eps in epses:
        for step in steps:
            run(eps, step, blurs, file)
    print(f'This process took: {(time.time() - start)/60} minutes')
    file.close()
            
if __name__ == '__main__':
    main()

