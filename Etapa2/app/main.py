# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from sklearn.metrics import precision_score, recall_score, f1_score
from app.utils.logging import setup_logging
import logging
from app.model.model import ModelHandler

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# Definir modelos de datos para las solicitudes y respuestas
class PredictionRequest(BaseModel):
    Textos_espanol: str  # Cambiado de List[str] a str

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

# Definir una ruta raíz opcional
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Predicción de ODS. Usa /docs para interactuar con la API."}

# Endpoint /predict modificado
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

# Endpoint /retrain sin cambios en el modelo de etiquetas, pero asegúrate de aceptar 3,4,5 directamente
@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest):
    global model_handler
    try:
        # Validar la longitud de los datos
        if len(request.Textos_espanol) != len(request.sdg):
            logger.error("Longitud de Textos_espanol y sdg no coinciden.")
            raise HTTPException(status_code=400, detail="La longitud de Textos_espanol y sdg debe ser igual.")

        # Validar que las etiquetas sean 3, 4 o 5
        for label in request.sdg:
            if label not in [3, 4, 5]:
                logger.error(f"Etiqueta sdg inválida: {label}")
                raise HTTPException(status_code=400, detail="Etiquetas sdg inválidas. Deben ser 3, 4 o 5.")

        # Reentrenar el modelo
        model_handler.retrain(request.Textos_espanol, request.sdg)

        # Evaluar el modelo
        y_pred = model_handler.model.predict(request.Textos_espanol)
        precision = precision_score(request.sdg, y_pred, average='weighted')
        recall = recall_score(request.sdg, y_pred, average='weighted')
        f1 = f1_score(request.sdg, y_pred, average='weighted')

        return RetrainResponse(precision=precision, recall=recall, f1_score=f1)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")
        raise HTTPException(status_code=500, detail="Error en el reentrenamiento del modelo.")
