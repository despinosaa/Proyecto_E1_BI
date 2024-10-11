from fastapi import FastAPI, HTTPException, File, UploadFile, Response
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
    

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if model_handler is None:
        logger.error("Modelo no cargado.")
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        if 'Textos_espanol' not in df.columns:
            raise HTTPException(status_code=400, detail="'Textos_espanol' column is required in the CSV file.")
        
        predictions_with_probabilities = model_handler.predict_with_probability_csv(df['Textos_espanol'].tolist())
        
        df['sdg'] = [pred for pred, _ in predictions_with_probabilities]
        df['probability'] = [prob for _, prob in predictions_with_probabilities]
        
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0) 
        
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail="Error en la predicción.")
