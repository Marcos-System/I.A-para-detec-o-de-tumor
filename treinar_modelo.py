import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib  # Importando joblib para salvar o modelo

# Carregar os dados
X = np.load('features.npy')
y = np.load('labels.npy')

# Dividir os dados em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Redução de dimensionalidade com PCA (mantendo 95% da variância)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Criar e treinar o modelo Random Forest
rf = RandomForestClassifier(random_state=42)

# Definir o grid de hiperparâmetros
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Usar GridSearchCV para otimizar os hiperparâmetros
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_pca, y_train)

# Melhor modelo
best_rf = grid_search.best_estimator_

# Previsões no conjunto de teste
y_pred = best_rf.predict(X_test_pca)

# Calcular a acurácias
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Salvar o modelo treinado
joblib.dump(best_rf, 'modelo_random_forest.joblib')  # Salvando o modelo

# Salvar o PCA
joblib.dump(pca, 'pca_random_forest.joblib')  # Salvando o PCA, caso queira usá-lo depois
