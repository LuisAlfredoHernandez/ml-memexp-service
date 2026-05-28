import joblib
import pandas as pd
from core.config import settings

class DeliveryTimePredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carga el modelo en memoria al iniciar el servidor."""
        try:
            self.model = joblib.load(settings.MODEL_PATH)
            print(f"✅ Modelo cargado correctamente desde: {settings.MODEL_PATH}")
        except FileNotFoundError:
            print(f"⚠️ Advertencia: No se encontró el modelo en {settings.MODEL_PATH}. Usando fallback matemático.")

    def predict(self, cantidad_piezas: int, prioridad_alta: bool, lineas_produccion: int) -> tuple[float, float]:
        if self.model is None:
            # Fallback (Plan B) por si corres la API sin haber entrenado el modelo aún
            base_horas = (cantidad_piezas / lineas_produccion) * 0.5
            return float(round(base_horas, 2)), 2.0

        # 1. Armamos el DataFrame exacto como lo espera scikit-learn
        df_entrada = pd.DataFrame([{
            "cantidad_piezas": cantidad_piezas,
            "prioridad_alta": 1 if prioridad_alta else 0,
            "lineas_produccion": lineas_produccion
        }])

        # 2. Ejecutamos la predicción
        estimacion = self.model.predict(df_entrada)[0]
        
        # Simulamos un margen de error (en un proyecto real lo calcularías con métricas como MAE/RMSE)
        margen_error = estimacion * 0.08 

        return float(round(estimacion, 2)), float(round(margen_error, 2))

# Instanciamos la clase para que el modelo se cargue una sola vez en la memoria
predictor = DeliveryTimePredictor()