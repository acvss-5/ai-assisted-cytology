import torch
from PIL import Image
from sklearn import svm
import torch
import torchvision.transforms as T


transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path, device='cuda'):
    # Open the image file
    img = Image.open(image_path).convert('RGB')
    
    # Apply the transformations
    img_tensor = transform(img)
    
    # Add batch dimension and move to the specified device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor
