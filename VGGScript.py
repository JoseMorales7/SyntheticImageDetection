# %%
import torchvision
from torchvision.models import vgg19
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import ImageData
import numpy as np
import json
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import time

# %%
model_name = "VGG"
model_image_size = 224
# vit = models.vit_l_16(models.ViT_L_16_Weights.IMAGENET1K_V1)

# %%
class VGG(torch.nn.Module):
    def __init__(self, numClasses):
        super(VGG, self).__init__()
        vgg = vgg19(weights = "DEFAULT")
        self.featureExtractor = vgg.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.class1, _, _, self.class2, _, _, _  = list(vgg.classifier.children())
        self.class3 = torch.nn.Linear(in_features = 4096, out_features = numClasses)
        self.classifier = torch.nn.Sequential(
            self.class1,
            torch.nn.ReLU(inplace=True),
            self.class2,
            torch.nn.ReLU(inplace=True),
            self.class3
        )

    def forward(self, x):
        #do something 
        x = self.featureExtractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

# %%
model = VGG(4).to(device)
# print(*list(model.children())[:-1])

# %%
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
batch_size = 32
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.RandomResizedCrop(size=(model_image_size, model_image_size), antialias=True), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainDataLoader, validDataLoader, testDataLoader, unseenDataLoader = ImageData.getImagesDataloaders("../ArtiFact/", transforms = transform, batchSize=batch_size)


def evaluate_on_data(model, dataloader, dirty: bool = False):
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        total_loss = 0
        
        num_correct = 0.0
        num_correct_dirty = 0.0

        num_samples = 0.0
        for data in tqdm(dataloader, desc="Eval: "):
            image, label = data
            label = label.to(device)
            image = image.to(device)
            outputs = model(image)
            
            dirtyLabel = torch.where(label > 1, torch.tensor(1, dtype = torch.int32).to(device), label)

            
            
            loss = criterion(outputs, label)
            total_loss += loss.item()
            argMax = torch.argmax(outputs, 1)

            # print("pred")
            # print(outputs)
            # print(argMax)
            # print("gt")
            # print(label)
            for i in range(len(label)):
                
                if label[i] == argMax[i]:
                    num_correct += 1

                if (dirtyLabel[i] == argMax[i]) or (dirtyLabel[i] == 1 and argMax[i] > 0):
                    num_correct_dirty += 1

                num_samples += 1
    return total_loss / len(dataloader), num_correct / num_samples, num_correct_dirty / num_samples

# %%
num_epochs = 10
count = 0
valid_loss_array = np.zeros(num_epochs)
valid_acc_array = np.zeros(num_epochs)
valid_acc_dirty_array = np.zeros(num_epochs)

train_loss_array = np.zeros(num_epochs)
for epoch in range(num_epochs):
    batch_count = 0
    for data in tqdm(trainDataLoader, desc="Training: "):
        
        image, label = data
        
        label = label.to(device)
        image = image.to(device)

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        count += 1
        # print(loss)
            
        
    valid_loss, valid_acc, valid_acc_dirty = evaluate_on_data(model, validDataLoader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Valid Loss: {valid_loss}, Valid ACC: {valid_acc}, Dirty Valid ACC: {valid_acc_dirty}')
    valid_loss_array[epoch] = valid_loss
    train_loss_array[epoch] = loss.item()
    valid_acc_array[epoch] = valid_acc
    valid_acc_dirty_array[epoch] = valid_acc_dirty

#9:53:40
#9:55:28


# %%
test_loss, test_acc, test_acc_dirty = evaluate_on_data(model, testDataLoader)
unseen_loss, unseen_acc, unseen_acc_dirty = evaluate_on_data(model, unseenDataLoader)

testingData = {
    "test_loss" : test_loss,
    "test_acc" : test_acc,
    "test_acc_dirty": test_acc_dirty,
    "unseen_loss" : unseen_loss, 
    "unseen_acc" : unseen_acc, 
    "unseen_acc_dirty": unseen_acc_dirty
}

# Serializing json
json_object = json.dumps(testingData, indent=4)
 
# Writing to sample.json
with open(f"./results/{model_name}_testing.json", "w") as outfile:
    outfile.write(json_object)

# %%
with open(f"./results/{model_name}_valid_loss.npy", 'wb') as f:
    np.save(f, valid_loss_array)
    
with open(f"./results/{model_name}_valid_acc.npy", 'wb') as f:
    np.save(f, valid_acc_array)

with open(f"./results/{model_name}_valid_dirty_acc.npy", 'wb') as f:
    np.save(f, valid_acc_dirty_array)
    
with open(f"./results/{model_name}_train.npy", 'wb') as f:
    np.save(f, train_loss_array)

torch.save(model.state_dict(), f"./savedModels/{model_name}Params.pth")


