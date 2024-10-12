import joblib # type: ignore
import logging
from typing import Tuple, List
import pandas as pd # type: ignore
from sklearn.metrics import f1_score  # type: ignore

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path=data_path
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
        
    def load_original_data(self) -> pd.DataFrame:
        try:
            if self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                df = pd.read_excel(self.data_path)
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                raise ValueError("Formato de archivo de datos no soportado.")
            logger.info(f"Datos originales cargados desde {self.data_path}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar los datos originales: {e}")
            raise e

    def predict(self, text: str) -> int:
        try:
            prediction = self.model.predict([text])[0]
            logger.info(f"Predicción realizada: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise e

    def predict_with_probabilities(self, text: str) -> Tuple[int, dict]:
        try:
            probabilities = self.model.predict_proba([text])[0]
            prediction = self.model.predict([text])[0]
            ods_probabilities = {
                3: probabilities[0],
                4: probabilities[1],
                5: probabilities[2]
            }
            logger.info(f"Predicción: {prediction} con probabilidades: {ods_probabilities}")
            return prediction, ods_probabilities
        except Exception as e:
            logger.error(f"Error en predicción con probabilidades: {e}")
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
                logger.info(f"Predicciones realizadas para {len(texts)} textos.")
                return predictions  

            except Exception as e:
                logger.error(f"Error en predicción con probabilidad: {e}")
                raise e
            
    def retrain(self, new_texts: List[str], new_labels: List[int]) -> Tuple[float, float]:
            try:
                # Cargar datos originales
                original_df = self.load_original_data()
                original_texts = original_df['Textos_espanol'].tolist()
                original_labels = original_df['sdg'].tolist()

                # Combinar datos originales con nuevos
                combined_texts = original_texts + new_texts
                combined_labels = original_labels + new_labels

                # Evaluar F1 Score antes del reentrenamiento
                y_pred_before = self.model.predict(original_texts)
                f1_before = f1_score(original_labels, y_pred_before, average='weighted')
                logger.info(f"F1 Score antes del reentrenamiento: {f1_before:.4f}")

                # Reentrenar el modelo con los datos combinados
                self.model.fit(combined_texts, combined_labels)
                self.save_model()
                logger.info("Modelo reentrenado con datos combinados.")

                # Evaluar F1 Score después del reentrenamiento
                y_pred_after = self.model.predict(original_texts)
                f1_after = f1_score(original_labels, y_pred_after, average='weighted')
                logger.info(f"F1 Score después del reentrenamiento: {f1_after:.4f}")

                return f1_before, f1_after

            except Exception as e:
                logger.error(f"Error en reentrenamiento: {e}")
                raise e