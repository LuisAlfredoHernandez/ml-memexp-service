from fastapi import FastAPI
from api.routes import router as predict_router
from core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Microservicio de IA para predicción de tiempos de producción",
    version="1.0.0"
)

# Incluimos las rutas de nuestra API
app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "mensaje": "Servicio de ML operativo",
        "documentacion": "Visita /docs para probar los endpoints"
    }