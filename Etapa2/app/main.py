from fastapi import FastAPI, HTTPException, File, UploadFile, Response, Request
import pandas as pd
from io import StringIO
from pydantic import BaseModel
from typing import List
from sklearn.metrics import precision_score, recall_score, f1_score
from app.utils.logging import setup_logging
import logging
from app.model.model import ModelHandler
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates

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
#ruta para los templates 
templates = Jinja2Templates(directory="app/templates") 
# Ruta al modelo
MODEL_PATH = 'models/model.joblib'

# Inicializar el manejador del modelo
try:
    model_handler = ModelHandler(MODEL_PATH)
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    model_handler = None

# Ruta raíz para servir el HTML
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("Navbar.html", {"request": request})
# Ruta para la página de predicción de texto
@app.get("/predict")
async def predict_text(request: Request):
    return templates.TemplateResponse("prediccion.html", {"request": request})
# Ruta para la página de predicción con CSV
@app.get("/predict_csv")
async def predict_csv(request: Request):
    return templates.TemplateResponse("prediccionCsv.html", {"request": request})
# Ruta para la página de reentrenamiento
@app.get("/retrain")
async def retrain(request: Request):
    return templates.TemplateResponse("reentrenar.html", {"request": request})

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