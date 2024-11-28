import cv2
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib

model = joblib.load('modelo_random_forest.joblib')  
pca = joblib.load('pca_random_forest.joblib') 

le = LabelEncoder()
le.fit(np.load('labels.npy')) 


def preprocess_and_extract_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64)) 
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray)
    image_normalized = equalized_image / 255.0
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def extract_hog_features(image_path):
    from skimage.feature import hog
    from skimage import exposure

    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features, hog_image = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return features

def classify_image(image_path):
    color_hist = preprocess_and_extract_histogram(image_path)
    hog_features = extract_hog_features(image_path)

    full_features = np.concatenate([color_hist, hog_features])
    full_features_pca = pca.transform([full_features])


    prediction = model.predict(full_features_pca)
    predicted_label = le.inverse_transform(prediction)
    if predicted_label == 0:
        print(f"Benigno")
    elif predicted_label == 1:
        print(f"maligno")
    else:
        print(f"normal")


image_path = "test/imagem9.png"  

classify_image(image_path)
