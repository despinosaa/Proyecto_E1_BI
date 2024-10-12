import joblib # type: ignore
import pandas as pd # type: ignore # type: ignore
from sklearn.pipeline import Pipeline # type: ignore # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from nltk.corpus import stopwords # type: ignore
import nltk # type: ignore
import os

nltk.download('stopwords')

def create_pipeline(max_features=8000, alpha=1.0):
    spanish_stopwords = stopwords.words('spanish')
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=spanish_stopwords, max_features=max_features)),
        ('clf', MultinomialNB(alpha=alpha, force_alpha=False))
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
    Y = df['sdg']
    
    pipeline = create_pipeline(max_features=max_features, alpha=alpha)
    pipeline.fit(X, Y)
    
    # Persistir el modelo
    joblib.dump(pipeline, model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    train_and_persist_model(
        data_path='data/ODScat_345.xlsx',
        model_path='models/model.joblib',
        max_features=8000,
        alpha=1.0
    )
