import torch # this is the main engine of the pytorch liabrary use overall function without it no operation can be perform

import torch.nn as nn # this is just liabrary of neural network it has relu, linear etc say this is the brain of the prohram 
from torchvision import datasets, transforms # Loeaded the dataset from torchvision and tranforms (make it into number 0- 1)
from torch.utils.data import DataLoader # this liabrary is use to load the data set in functional way like shuffle the data , import multiple value of the data at a time into the program 
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([ #Apply multiple transformations one after another.
    transforms.Grayscale(), # transform image into black and white becpz some images may have different channels (colors)like (1,28,24) and more it will crash the program 
    transforms.ToTensor(), # Converst the imahges into numbers (0-1)
    transforms.Normalize((0.5,),(0.5,))
])   

dataset = datasets.ImageFolder(
    root="mnist_png/mnist_png/training",
    transform=transform   # this is the secound transform function keeping evertyhi show above in data set also so function are directly getting perform in it 
)



loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    
) # it takes all the 64 images and shuffles it so model doesnt memorizes it 


# Torch is the engine that lets learn the computer about neural network
class Descriminator(nn.Module):
    def __init__(self):           # it actully initilze the parent class nn.module()
        super().__init__()
        self.net = nn.Sequential( #Creates a pipeline of layers, Data flows through them in order Input → Layer 1 → Layer 2 → Layer 3 → Output
            nn.Linear(784,256),  # takes the 784 as input and 256 as output 28 × 28 = 784 converts image pixels into useful features 
            nn.LeakyReLU(0.2),  # Adds non-linearity, Prevents neurons from “dying”
            nn.Linear(256,1), # takes 256 features and output 1
            nn.Sigmoid() # Squashes output to range 0–1
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),     # noise → features
            nn.ReLU(),
            nn.Linear(256, 784),     # features → image
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

D = Descriminator().to(device)
G = Generator().to(device)

criterion = nn.BCELoss()

lr = 0.0002 # learning speed (too big → unstable, too small → slow)

optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)   
# adam is smart weight updater 

epochs = 80  # keep 1 for now (testing)

for epoch in range(epochs):
    for real_imgs, _ in loader:

        # -------- Train Discriminator --------
        real_imgs = real_imgs.view(-1, 784).to(device)

        real_labels = torch.ones(real_imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(real_imgs.size(0), 1).to(device)

        noise = torch.randn(real_imgs.size(0), 100).to(device)
        fake_imgs = G(noise)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())

        loss_real = criterion(D_real, real_labels)
        loss_fake = criterion(D_fake, fake_labels)

        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # -------- Train Generator --------
        output = D(fake_imgs)
        loss_G = criterion(output, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")


matplotlib.use("TkAgg")

G.eval() # switch generater to evaluation mode 

noise = torch.randn(16,100).to(device)
fake_imgs = G(noise)

fake_imgs = fake_imgs.view(-1, 28, 28).detach().cpu().numpy()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_imgs[i], cmap="gray")
    ax.axis("off")

plt.show()


