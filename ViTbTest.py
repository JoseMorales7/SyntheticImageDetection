# %%
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import ImageData
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
import json

from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# %%
model_name = "ViT_b_16_10E"
model_image_size = 224
# vit = models.vit_l_16(models.ViT_L_16_Weights.IMAGENET1K_V1)

# %%
class ViT(torch.nn.Module):
    def __init__(self, numClasses: int):
        super(ViT, self).__init__()
        
        self.reference_vit = torchvision.models.vit_b_16(torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.reference_vit.heads.head = torch.nn.Sequential(torch.nn.Linear(768, 512),
                                                            torch.nn.ReLU(),
                                                            torch.nn.Linear(512, 256),
                                                            torch.nn.ReLU(),
                                                            torch.nn.Linear(256, numClasses))
        self.reference_vit.conv_proj.requires_grad = False
        self.reference_vit.encoder.requires_grad = False
        self.reference_vit.heads.requires_grad = False
        self.reference_vit.heads.head.requires_grad = True

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.reference_vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.reference_vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x =  self.reference_vit.heads(x)
        
        #extractedFeature = self.ViT(x)
        softmax = self.softmax(x)

        return softmax
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.reference_vit.patch_size
        torch._assert(h == self.reference_vit.image_size, f"Wrong image height! Expected {self.reference_vit.image_size} but got {h}!")
        torch._assert(w == self.reference_vit.image_size, f"Wrong image width! Expected {self.reference_vit.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x =  self.reference_vit.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.reference_vit.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

# %%
model = ViT(4).to(device)
# print(*list(model.children())[:-1])

# %%
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# %%
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.RandomResizedCrop(size=(model_image_size, model_image_size), antialias=True), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainDataLoader, validDataLoader, testDataLoader, unseenDataLoader = ImageData.getImagesDataloaders(r"../ArtiFact", transforms = transform, batchSize=batch_size)

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


# dataset = trainDataLoader.dataset
# trainIdxs, _ = train_test_split(range(len(dataset.imagePaths)), train_size=500, random_state=42)
# ValIdxs = trainIdxs[490:]
# trainIdxs = trainIdxs[:490]
# trainSub = Subset(dataset, trainIdxs)
# valSub = Subset(dataset, ValIdxs)
# trainSubDataloader = DataLoader(trainSub, batch_size=32, shuffle=True)
# valSubDataloader = DataLoader(valSub, batch_size=32, shuffle=True)

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

torch.save(model.state_dict(), "./savedModels/" + model_name + "Params.pth")
