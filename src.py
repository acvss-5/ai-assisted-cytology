import torch
import numpy as np
from img_loader import load_image
import joblib

svm_path = "models\svm_model.joblib"
knn_path = "models\knn_model.joblib"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.to(device)
dinov2.eval()

def img_embeddings(img):
    img = load_image(image_path=img)

    with torch.no_grad():
        embedding = dinov2(img.to(device))
        
        # Reshape the embedding to a 2D array
        embedding_2d = np.array(embedding[0].cpu()).reshape(1, -1)
        
    return embedding_2d


def classify(embeddings):
    if "svm":
        # Load the model from the file
        clf = joblib.load(svm_path)

        # Now you can use the loaded classifier to make predictions
        predictions = clf.predict(embeddings.cpu().numpy())[0]

        return predictions
    
