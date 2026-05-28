from fastapi import APIRouter
from api.schemas import PredictionRequest, PredictionResponse
from services.prediction import predictor

router = APIRouter()

@router.post("/predict/delivery-time", response_model=PredictionResponse)
async def predict_delivery_time(request: PredictionRequest):
    
    # Llamamos al servicio de ML para obtener el cálculo
    tiempo, error = predictor.predict(
        cantidad_piezas=request.cantidad_piezas,
        prioridad_alta=request.prioridad_alta,
        lineas_produccion=request.lineas_produccion
    )
    
    return PredictionResponse(
        tiempo_estimado_horas=tiempo,
        margen_error_horas=error,
        modelo_version="random_forest_v1"
    )