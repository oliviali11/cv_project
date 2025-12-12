import timm
import torch
from autoattack import AutoAttack
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

model = timm.create_model('resnet50.a1_in1k', pretrained = True)
model.eval()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

valdir = '/cs/cs153/projects/olivia-elsa/cv_final_project/ImageNet_final_val_sorted'

dataset = datasets.ImageFolder(
    valdir,
    transform=T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    allow_empty=True
)

val_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)

epsilon = 0.031

clean_acc = []
adv_acc = []
attacks_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
attack_acc = []

avg_clean_acc = []

num_batches = 25 # default

for k in range(4):
    print(f"START ATTACK FOR {attacks_list[k]}")
    for i, (images, labels) in enumerate(val_loader):

        if i >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)

        batch_size = labels.size(0)

        # Clean accuracy for this batch
        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(1)
            clean_correct = (preds == labels).sum().item()
        
        # total_seen += labels.size(0)
        clean_acc.append((float) (clean_correct / batch_size))

        adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='custom', attacks_to_run=[attacks_list[k]], device=device, verbose = False)

        adv_images = adversary.run_standard_evaluation(images, labels, bs=batch_size)

        # robust accuracy for this batch
        with torch.no_grad():
            outputs_adv = model(adv_images)
            preds_adv = outputs_adv.argmax(1)
            adv_correct = (preds_adv == labels).sum().item()
        
        adv_acc.append((float) (adv_correct / batch_size))

        # Print running accuracy
        print(f"Batch {i+1}/{len(val_loader)}: Clean accuracy = {clean_acc[i]}, Robust accuracy (adv) = {adv_acc[i]}")

    attack_acc.append(np.mean(adv_acc))
    avg_clean_acc.append(np.mean(clean_acc))

    clean_acc = []
    adv_acc = []

print(f"Overall Clean Accuracy = {avg_clean_acc[0]}")
print(f"Robust accuracy for apgd-ce, apgd-t, fab-t, square = {attack_acc}")


