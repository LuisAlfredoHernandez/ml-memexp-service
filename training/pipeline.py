import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.config import settings
from core.database import SessionLocal

def train_model():
    """
    Ejecuta el pipeline de reentrenamiento del modelo utilizando datos reales de la BD.
    Cumple con:
    - RF19 (Reentrenamiento manual)
    - RF20 / RNF-09 (Validar suficiencia de datos de al menos 15 días)
    - RF21 / RNF-10 (Comparar MAE/MSE y bloquear publicación si hay degradación)
    - RNF-03 (Acceso a BD de forma segura de solo lectura)
    """
    db = SessionLocal()
    try:
        # 1. RNF-09: Validar que existan al menos 15 días de registros de producción en la BD
        dias_query = text("""
            SELECT COUNT(DISTINCT DATE(fecha_reporte)) 
            FROM reporte_avance 
            WHERE estado = 'validado'
        """)
        unique_days = db.execute(dias_query).scalar() or 0
        
        if unique_days < 15:
            raise ValueError(
                f"Suficiencia de datos inválida: Se requieren registros de producción distribuidos "
                f"en al menos 15 días únicos para entrenar (actualmente hay {unique_days} días)."
            )

        # 2. RNF-03 / RF5: Consultar histórico de producción (Solo Lectura)
        query = text("""
            SELECT 
                ao.piezas_requeridas AS cantidad_piezas,
                CASE WHEN LOWER(o.prioridad::text) IN ('alta', 'urgente') THEN 1 ELSE 0 END AS prioridad_alta,
                1 AS lineas_produccion,
                EXTRACT(EPOCH FROM (MAX(ra.fecha_reporte) - ao.fecha_asignacion)) / 3600.0 AS tiempo_horas
            FROM asignacion_orden ao
            JOIN orden o ON ao.orden_id = o.id
            JOIN reporte_avance ra ON ra.asignacion_id = ao.id
            WHERE ao.estado::text = 'COMPLETADA' AND ra.estado = 'validado'
            GROUP BY ao.id, ao.piezas_requeridas, o.prioridad, ao.fecha_asignacion
        """)
        
        result = db.execute(query).fetchall()
        
        if len(result) < 10:
            # Si hay 15 días de reportes pero no hay suficientes asignaciones finalizadas para entrenar ML
            raise ValueError(
                f"Registros insuficientes: Se necesitan al menos 10 órdenes finalizadas "
                f"para calibrar las predicciones (actualmente hay {len(result)})."
            )

        # 3. Modelado con Pandas & Scikit-learn
        df = pd.DataFrame(result, columns=["cantidad_piezas", "prioridad_alta", "lineas_produccion", "tiempo_horas"])
        
        X = df[["cantidad_piezas", "prioridad_alta", "lineas_produccion"]]
        y = df["tiempo_horas"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar RandomForest
        new_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        new_model.fit(X_train, y_train)

        # Calcular métricas de error
        y_pred = new_model.predict(X_test)
        new_mae = float(mean_absolute_error(y_test, y_pred))
        new_mse = float(mean_squared_error(y_test, y_pred))

        # 4. RF21 / RNF-10: Comparar con el modelo activo si existe
        active_mae = None
        active_mse = None
        
        if os.path.exists(settings.MODEL_PATH):
            try:
                active_model = joblib.load(settings.MODEL_PATH)
                y_pred_active = active_model.predict(X_test)
                active_mae = float(mean_absolute_error(y_test, y_pred_active))
                active_mse = float(mean_squared_error(y_test, y_pred_active))
                
                # Validar degradación: si el nuevo MAE es mayor al del modelo actual, cancelar
                if new_mae > active_mae:
                    raise ValueError(
                        f"Degradación de precisión detectada. "
                        f"El error MAE del nuevo modelo ({new_mae:.3f} hrs) supera "
                        f"al del modelo actual en producción ({active_mae:.3f} hrs)."
                    )
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
                # Si falla al cargar el modelo anterior, asumimos que no hay modelo válido previo
                pass

        # 5. Guardar artefactos
        artifacts_dir = os.path.dirname(settings.MODEL_PATH)
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(new_model, settings.MODEL_PATH)

        return {
            "estado": "exitoso",
            "registros_entrenados": len(df),
            "mae_actual": active_mae,
            "mse_actual": active_mse,
            "mae_nuevo": new_mae,
            "mse_nuevo": new_mse,
            "version_publicada": "random_forest_v1"
        }
        
    finally:
        db.close()
