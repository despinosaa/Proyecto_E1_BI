from fastapi import FastAPI, HTTPException, File, UploadFile, Response, Request # type: ignore
import pandas as pd # type: ignore # type: ignore
from io import StringIO
from pydantic import BaseModel # type: ignore # type: ignore
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score # type: ignore # type: ignore
from app.utils.logging import setup_logging
import logging
from app.model.model import ModelHandler
from fastapi.responses import HTMLResponse  # type: ignore # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore # type: ignore

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# Definir modelos de datos para las solicitudes y respuestas
class PredictionRequest(BaseModel):
    Textos_espanol: str

class PredictionResponse(BaseModel):
    ods: int
    probabilities: Dict[int, float]

class RetrainRequest(BaseModel):
    Textos_espanol: List[str]
    sdg: List[int]

class RetrainResponse(BaseModel):

    f1_score_before: float
    f1_score_after: float

# Inicializar la aplicación
app = FastAPI(title="ODS Prediction API", description="API para predecir niveles de ODS basado en texto.")

# Ruta para los templates 
templates = Jinja2Templates(directory="app/templates") 

# Ruta al modelo
MODEL_PATH = 'models/model.joblib'
DATA_PATH = 'data/ODScat_345.xlsx'

# Inicializar el manejador del modelo
try:
    model_handler = ModelHandler(MODEL_PATH, DATA_PATH)
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    model_handler = None

# Ruta raíz para servir el HTML
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("inicio.html", {"request": request})

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
        ods_prediction, probabilities = model_handler.predict_with_probabilities(request.Textos_espanol)
        return PredictionResponse(ods=ods_prediction, probabilities=probabilities)
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
        df_new = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validar la existencia de columnas necesarias
        if 'Textos_espanol' not in df_new.columns or 'sdg' not in df_new.columns:
            logger.error("El CSV debe contener las columnas 'Textos_espanol' y 'sdg'.")
            raise HTTPException(status_code=400, detail="El CSV debe contener las columnas 'Textos_espanol' y 'sdg'.")
        
        # Extraer los textos y las etiquetas
        texts_new = df_new['Textos_espanol'].tolist()
        labels_new = df_new['sdg'].tolist()

        # Validar la longitud de los datos
        if len(texts_new) != len(labels_new):
            logger.error("Longitud de Textos_espanol y sdg no coinciden.")
            raise HTTPException(status_code=400, detail="La longitud de Textos_espanol y sdg debe ser igual.")

        # Validar que las etiquetas sean 3, 4 o 5
        for label in labels_new:
            if label not in [3, 4, 5]:
                logger.error(f"Etiqueta sdg inválida: {label}")
                raise HTTPException(status_code=400, detail="Etiquetas sdg inválidas. Deben ser 3, 4 o 5.")

        # Reentrenar el modelo y obtener F1 Scores
        f1_before, f1_after = model_handler.retrain(texts_new, labels_new)

        return RetrainResponse(f1_score_before=f1_before, f1_score_after=f1_after)
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")
        raise HTTPException(status_code=500, detail="Error en el reentrenamiento del modelo.")