import joblib
import logging
from typing import Tuple, List

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

    def predict(self, text: str) -> int:
        try:
            prediction = self.model.predict([text])[0]
            logger.info(f"Predicción realizada: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise e

    def predict_with_probability(self, text: str) -> Tuple[int, float]:
        try:
            prediction = self.model.predict([text])[0]
            probabilities = self.model.predict_proba([text])[0]
            max_prob = round(probabilities.max(), 4)
            logger.info(f"Predicción: {prediction} con probabilidad: {max_prob}")
            return prediction, max_prob
        except Exception as e:
            logger.error(f"Error en predicción con probabilidad: {e}")
            raise e

    def retrain(self, texts: list, labels: list):
        try:
            self.model.fit(texts, labels)
            logger.info("Modelo reentrenado con nuevos datos.")
            self.save_model()
        except Exception as e:
            logger.error(f"Error en reentrenamiento: {e}")
            raise e


    def predict_with_probability_csv(self, texts: List[str]) -> List[Tuple[int, float]]:
            predictions = []
            
            try:
                for text in texts:
                    prediction = self.model.predict([text])[0]
                    probabilities = self.model.predict_proba([text])[0]
                    max_prob = round(probabilities.max(), 4)
                    logger.info(f"Predicción: {prediction} con probabilidad: {max_prob}")
                    predictions.append((prediction, max_prob))
            
                return predictions  

            except Exception as e:
                logger.error(f"Error en predicción con probabilidad: {e}")
                raise e