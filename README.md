# Developing-a-Convolutional-Neural-Network-Classification-Model
## Aim
To develop and implement a Convolutional Neural Network (CNN) classification model using PyTorch to accurately classify handwritten digits from the MNIST dataset.
## Theory
The MNIST (Modified National Institute of Standards and Technology) dataset is a foundational benchmark in computer vision. It consists of 70,000 grayscale images of handwritten digits ($0$ through $9$).Image Dimensions: $28 \times 28$ pixels (single channel).Dataset Split: 60,000 training images and 10,000 testing images.
### Why CNNs?
Convolutional Neural Networks are uniquely engineered for image processing tasks. Unlike traditional fully connected neural networks that discard spatial configuration by flattening inputs, CNNs utilize Convolutional layers to preserve spatial context and automatically extract features (such as edges, textures, and shapes) through shared weights and local receptive fields. Pooling layers reduce spatial dimensionality to ensure translation invariance and reduce computational overhead.
## Design Steps
### Step 1: Library Environment Setup
Import essential libraries including torch for core deep learning functionality, torchvision for data management, matplotlib for generating visual analytics, and sklearn for metric reporting.
### Step 2: Data Acquisition & Pipeline Configuration
Download the MNIST train and test sets. Configure data pipelines using Transforms.ToTensor() to map pixel values to a range of $[0.0, 1.0]$. Wrap datasets inside DataLoader structures with an optimal batch size (e.g., batch_size=10).
### Step 3: Model Architecture Formulation
Design the network using an OOP approach by extending nn.Module. Chain together two convolutional layers interleaved with Max-Pooling operations, flatten the dimensional volume, and feed it through deep linear layers down to the output vector.
### Step 4: Optimization Setup
Declare the loss criteria using Cross-Entropy Loss (nn.CrossEntropyLoss). Set up the optimization algorithm using Adam (optim.Adam) with an initial learning rate set to $\alpha = 0.001$.
### Step 5: Training and Evaluation Loop
Iterate over 5 complete epochs. In each epoch, perform forward propagation, calculate training error, run backpropagation (loss.backward()), and optimize the weight parameters (optimizer.step()). Track loss profiles and evaluation metrics along the timeline.
### Step 6: Post-Training Diagnostics
Generate metric graphics comparing training loss against validation loss, plot accuracy over time, and output a detailed Multi-Class Confusion Matrix across the 10,000-sample test cohort.
## Program
```
Developed by :Prajin S
Register Number  212223230151
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)
train_data
test_data
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
torch.manual_seed(42)
model = ConvolutionalNetwork()
model
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
import time
start_time = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        
        # Apply the model
        y_pred = model(X_train)  # we not flatten X-train here
        loss = criterion(y_pred, y_train)
 
        
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()  # Trure 1 / False 0 sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print interim results
        if b%600 == 0:
            print(f'epoch: {i}  batch: {b} loss: {loss.item()}')
        
    train_losses.append(loss)
    train_correct.append(trn_corr)
        
    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()
            
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)
        
current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')
train_losses = [t.detach().numpy() for t in train_losses]
test_losses = [t.detach().numpy() for t in test_losses]

plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend();
plt.show()
plt.plot([t/600 for t in train_correct], label='training accuracy')
plt.plot([t/100 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)  # we don't flatten the data this time
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
print(np.arange(10).reshape(1,10))
print()
print(confusion_matrix(predicted.view(-1), y_test.view(-1)))
model.eval()
with torch.no_grad():
    new_prediction = model(test_data[2019][0].view(1,1,28,28))
new_prediction.argmax()
plt.imshow(test_data[334][0].reshape(28,28))
plt.show()
```
## OUTPUT
### Model
<img width="559" height="159" alt="image" src="https://github.com/user-attachments/assets/56805270-6ea9-4fa5-91af-69587d3daaf7" />

### Training Loss per Epoch
<img width="370" height="798" alt="image" src="https://github.com/user-attachments/assets/2865b80e-e3e9-4804-9866-f49413a2d675" />

<img width="358" height="21" alt="image" src="https://github.com/user-attachments/assets/222b7503-8854-4555-bf48-ef45372e49f1" />

### Visualization Graph
<img width="774" height="538" alt="image" src="https://github.com/user-attachments/assets/fd91dcb8-fcbf-48de-9c88-5c6b164f2eaf" />

<img width="752" height="563" alt="image" src="https://github.com/user-attachments/assets/4cd208af-ca21-44fa-94f7-84ab3302ae27" />


### Confusion Matrix
<img width="570" height="267" alt="image" src="https://github.com/user-attachments/assets/6bcfaf21-4d98-47b9-8d03-18df1cdb7811" />

### New Sample Data Prediction
<img width="552" height="525" alt="image" src="https://github.com/user-attachments/assets/f635046c-dfcf-4ca7-b063-5743d5c05544" />



## RESULT
Thus the CNN model was trained and tested successfully on the MNIST dataset.
