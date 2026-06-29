from pydantic import BaseModel, Field
from typing import Optional, Any

# --- Petición de tiempo de entrega (RF12) ---
class PredictionRequest(BaseModel):
    cantidad_piezas: int = Field(..., gt=0, description="Cantidad total de piezas a fabricar")
    prioridad_alta: bool = Field(default=False, description="¿Es un pedido urgente?")
    lineas_produccion: int = Field(default=1, gt=0, description="Líneas de ensamblaje asignadas")

class PredictionResponse(BaseModel):
    tiempo_estimado_horas: float
    margen_error_horas: float
    modelo_version: str

# --- Petición de Simulación MTS (RF16) ---
class MtsSimulationRequest(BaseModel):
    cantidad_piezas: int = Field(..., gt=0, description="Cantidad total de la orden de stock (MTS)")

class MtsSimulationItem(BaseModel):
    orden: str
    antes: str
    despues: str
    impacto: str
    color: str

# --- Gestión / Reentrenamiento (RF19–RF22) ---
class TrainResponse(BaseModel):
    estado: str
    registros_entrenados: int
    mae_actual: Optional[float] = None
    mse_actual: Optional[float] = None
    mae_nuevo: float
    mse_nuevo: float
    version_publicada: str

# --- Seed Helper ---
class SeedResponse(BaseModel):
    estado: str
    mensaje: str
    registros_insertados: int