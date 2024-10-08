import joblib
import logging
from typing import List

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model()
    
    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            logger.info(f"Modelo cargado desde {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise e
    
    def save_model(self):
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Modelo guardado en {self.model_path}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")
            raise e
    
    def predict(self, texts: List[str]) -> List[int]:
        try:
            predictions = self.model.predict(texts)
            logger.info(f"Predicciones realizadas para {len(texts)} instancias.")
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise e
    
    def retrain(self, texts: List[str], labels: List[int]):
        try:
            self.model.fit(texts, labels)
            logger.info("Modelo reentrenado con nuevos datos.")
            self.save_model()
        except Exception as e:
            logger.error(f"Error en reentrenamiento: {e}")
            raise e
