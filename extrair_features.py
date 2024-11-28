import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage import exposure

# Função para extrair o histograma de cores com pré-processamento
def preprocess_and_extract_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Reduzir para uma resolução menor
    
    # Equalizar o histograma (melhora o contraste)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray)
    
    # Normalizar a imagem
    image_normalized = equalized_image / 255.0
    
    # Converter para o espaço de cores HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

# Função para extrair características HOG
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Reduzir para uma resolução menor
    # Converter a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extrair as características HOG
    features, hog_image = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    # Melhorar a visualização do HOG
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return features

# Definir o diretório de imagens (agora com subpastas para cada classe)
image_dir = 'Images'  # Substitua pelo caminho do seu dataset
labels = []
features = []

# Ler imagens e extrair características
for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)
    
    # Verificar se é uma pasta (classe)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            print(f"Processando imagem: {image_path}")
            
            # Extrair o histograma de cores e HOG
            color_hist = preprocess_and_extract_histogram(image_path)
            hog_features = extract_hog_features(image_path)
            
            # Concatenar as duas características
            full_features = np.concatenate([color_hist, hog_features])
            
            features.append(full_features)
            labels.append(class_name)  # O nome da pasta é o rótulo da classe

# Converter rótulos para números
le = LabelEncoder()
labels = le.fit_transform(labels)

# Converter a lista de características para um array numpy
features = np.array(features)
labels = np.array(labels)

# Salvar os dados para o modelo
np.save('features.npy', features)
np.save('labels.npy', labels)
