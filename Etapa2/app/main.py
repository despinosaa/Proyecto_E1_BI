from fastapi import FastAPI, HTTPException, File, UploadFile
import pandas as pd
from io import StringIO
from pydantic import BaseModel
from typing import List
from sklearn.metrics import precision_score, recall_score, f1_score
from app.utils.logging import setup_logging
import logging
from app.model.model import ModelHandler

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# Definir modelos de datos para las solicitudes y respuestas
class PredictionRequest(BaseModel):
    Textos_espanol: str

class PredictionResponse(BaseModel):
    ods: int
    probability: float

class RetrainRequest(BaseModel):
    Textos_espanol: List[str]
    sdg: List[int]

class RetrainResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float

# Inicializar la aplicación
app = FastAPI(title="ODS Prediction API", description="API para predecir niveles de ODS basado en texto.")

# Ruta al modelo
MODEL_PATH = 'models/model.joblib'

# Inicializar el manejador del modelo
try:
    model_handler = ModelHandler(MODEL_PATH)
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    model_handler = None

# Definir una ruta raíz
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Predicción de ODS. Usa /docs para interactuar con la API."}

# Endpoint /predict
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model_handler is None:
        logger.error("Modelo no cargado.")
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    try:
        ods_prediction, probability = model_handler.predict_with_probability(request.Textos_espanol)
        return PredictionResponse(ods=ods_prediction, probability=probability)
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail="Error en la predicción.")



@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)):
    global model_handler
    try:
        # Leer el archivo CSV
        contents = await file.read()
        # Use StringIO to convert the contents to a file-like object
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validar la existencia de columnas necesarias
        if 'Textos_espanol' not in df.columns or 'sdg' not in df.columns:
            logger.error("El CSV debe contener las columnas 'Textos_espanol' y 'sdg'.")
            raise HTTPException(status_code=400, detail="El CSV debe contener las columnas 'Textos_espanol' y 'sdg'.")
        
        # Extraer los textos y las etiquetas
        texts = df['Textos_espanol'].tolist()
        labels = df['sdg'].tolist()

        # Validar la longitud de los datos
        if len(texts) != len(labels):
            logger.error("Longitud de Textos_espanol y sdg no coinciden.")
            raise HTTPException(status_code=400, detail="La longitud de Textos_espanol y sdg debe ser igual.")

        # Validar que las etiquetas sean 3, 4 o 5
        for label in labels:
            if label not in [3, 4, 5]:
                logger.error(f"Etiqueta sdg inválida: {label}")
                raise HTTPException(status_code=400, detail="Etiquetas sdg inválidas. Deben ser 3, 4 o 5.")

        # Reentrenar el modelo
        model_handler.retrain(texts, labels)

        # Evaluar el modelo
        y_pred = model_handler.model.predict(texts)
        precision = precision_score(labels, y_pred, average='weighted')
        recall = recall_score(labels, y_pred, average='weighted')
        f1 = f1_score(labels, y_pred, average='weighted')

        return RetrainResponse(precision=precision, recall=recall, f1_score=f1)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")
        raise HTTPException(status_code=500, detail="Error en el reentrenamiento del modelo.")