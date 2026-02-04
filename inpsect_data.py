from torchvision import datasets, transforms
transform = transforms.Compose([ #Apply multiple transformations one after another.
    transforms.Grayscale(), # tyransform image into black and white becpz some images may have different channels (colors)like (1,28,24) and more it will crash the program 
    transforms.ToTensor(), # Converst the imahges into numbers 
    transforms.Normalize((0.5,),(0.5,))
])   

dataset = datasets.ImageFolder(
    root="mnist_png/mnist_png/training",
    transform=transform
)

img, lable = dataset[0]
# print(img.shape)
# print(lable)
