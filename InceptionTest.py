# %%
from torchvision.models import inception_v3
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import ImageData
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time

# %%
model_name = "Inception"
model_image_size = 299
# vit = models.vit_l_16(models.ViT_L_16_Weights.IMAGENET1K_V1)

# %%
class InceptionNet(torch.nn.Module):
    def __init__(self, numClasses: int):
        super(InceptionNet, self).__init__()

        self.inceptionBase = inception_v3(weights='DEFAULT')
        self.inceptionBase.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 216),
            torch.nn.ReLU(),
            torch.nn.Linear(216, numClasses)
        )
        
        for param in list(self.inceptionBase.parameters())[:-1]:
            param.requires_grad = True
        # for param in self.inceptionBase.parameters():
        #     print(param.requires_grad)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.inceptionBase(x)
        probs = self.softmax(logits.logits)

        return probs

# %%
model = InceptionNet(4).to(device)
# print(*list(model.children())[:-1])

# %%
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# %%
batch_size = 16
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.RandomResizedCrop(size=(model_image_size, model_image_size), antialias=True), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainDataLoader, validDataLoader, testDataLoader = ImageData.getImagesDataloaders("../ArtiFact/", transforms = transform, batchSize=batch_size)

# %%
dataset = trainDataLoader.dataset

# %%
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# %%
trainIdxs, _ = train_test_split(range(len(dataset.imagePaths)), train_size=110, random_state=42)

# %%
ValIdxs = trainIdxs[100:]
trainIdxs = trainIdxs[:100]

# %%
len(trainIdxs)

# %%
trainSub = Subset(dataset, trainIdxs)
valSub = Subset(dataset, ValIdxs)

# %%
trainSubDataloader = DataLoader(trainSub, batch_size=32, shuffle=True)
valSubDataloader = DataLoader(valSub, batch_size=32, shuffle=True)

# %%
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
    valid_acc_dirty_array[epoch] = valid_acc

#9:53:40
#9:55:28


# %%
with open(model_name + '_valid_loss.npy', 'wb') as f:
    np.save(f, valid_loss_array)
    
with open(model_name + '_valid_acc.npy', 'wb') as f:
    np.save(f, valid_acc_array)

with open(model_name + '_valid_dirty_acc.npy', 'wb') as f:
    np.save(f, valid_acc_dirty_array)
    
with open(model_name + '_train.npy', 'wb') as f:
    np.save(f, train_loss_array)

# %%

torch.save(model.state_dict(), f"./savedModels/{model_name}.pth")

