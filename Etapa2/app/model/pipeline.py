import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk
import os

# Descargar las stopwords de NLTK si no están disponibles
nltk.download('stopwords')

def create_pipeline(max_features=8000, alpha=1.0):
    spanish_stopwords = stopwords.words('spanish')
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=spanish_stopwords, max_features=max_features)),
        ('clf', MultinomialNB(alpha=alpha))
    ])
    
    return pipeline

def train_and_persist_model(data_path: str, model_path: str, max_features=8000, alpha=1.0):
    # Cargar datos
    df = pd.read_excel(data_path)
    
    # Preprocesamiento
    df['Textos_espanol'] = df['Textos_espanol'].str.lower()
    df['Textos_espanol'] = df['Textos_espanol'].str.replace('[^\w\s]', '', regex=True)
    
    # Vectorización y modelo
    X = df['Textos_espanol']
    Y = df['sdg'].map({3: 0, 4: 1, 5: 2})
    
    pipeline = create_pipeline(max_features=max_features, alpha=alpha)
    pipeline.fit(X, Y)
    
    # Persistir el modelo
    joblib.dump(pipeline, model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    # Asegurarse de que el directorio 'models/' exista
    os.makedirs('models', exist_ok=True)
    
    train_and_persist_model(
        data_path='data/ODScat_345.xlsx',
        model_path='models/model.joblib',
        max_features=8000,
        alpha=1.0
    )
