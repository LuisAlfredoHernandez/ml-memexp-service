from pydantic import BaseModel, Field

# Lo que recibimos desde tu backend en Next.js/FastAPI
class PredictionRequest(BaseModel):
    cantidad_piezas: int = Field(..., gt=0, description="Cantidad total de piezas a fabricar")
    prioridad_alta: bool = Field(default=False, description="¿Es un pedido urgente?")
    lineas_produccion: int = Field(default=1, gt=0, description="Líneas de ensamblaje asignadas")

# Lo que le devolvemos a tu backend
class PredictionResponse(BaseModel):
    tiempo_estimado_horas: float
    margen_error_horas: float
    modelo_version: str