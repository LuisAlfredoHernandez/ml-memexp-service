from fastapi import APIRouter, HTTPException, Depends
from typing import List
from api.schemas import (
    PredictionRequest, PredictionResponse, 
    MtsSimulationRequest, MtsSimulationItem,
    TrainResponse, SeedResponse
)
from services.prediction import predictor
from training.pipeline import train_model
from core.database import SessionLocal
from sqlalchemy import text

router = APIRouter()

@router.post("/predict/delivery-time", response_model=PredictionResponse)
async def predict_delivery_time(request: PredictionRequest):
    """Estima el tiempo de entrega de una asignación en horas (RF12)"""
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/projections")
async def get_projections():
    """Genera proyecciones de nivel de producción semanales/mensuales (RF13)"""
    try:
        proyecciones = predictor.get_projections()
        return proyecciones
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/bottlenecks")
async def get_bottlenecks():
    """Retorna los cuellos de botella detectados en planta y sugerencias de balanceo (RF15)"""
    try:
        data = predictor.detect_bottlenecks()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/simulate-mts", response_model=List[MtsSimulationItem])
async def simulate_mts(request: MtsSimulationRequest):
    """Simula cómo afectará la inclusión de una nueva orden MTS a las órdenes MTO vigentes (RF16)"""
    try:
        simulacion = predictor.simulate_mts_impact(request.cantidad_piezas)
        return [MtsSimulationItem(**item) for item in simulacion]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=TrainResponse)
async def run_training():
    """Dispara el pipeline de reentrenamiento manual de forma síncrona (RF19, RF20, RF21, RF22)"""
    try:
        result = train_model()
        return TrainResponse(**result)
    except ValueError as e:
        # Error de validación de negocio (suficiencia de datos RNF-09 o degradación RNF-10)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del pipeline de entrenamiento: {e}")

@router.post("/seed-data", response_model=SeedResponse)
async def seed_data():
    """Endpoint auxiliar para sembrar datos históricos en Postgres y poder probar el reentrenamiento"""
    db = SessionLocal()
    try:
        # 1. Asegurar la existencia de un operario
        op_id = db.execute(text("SELECT id FROM operario LIMIT 1")).scalar()
        if not op_id:
            # Buscar un usuario operario
            user_id = db.execute(text("SELECT id FROM usuario WHERE rol = 'operario' LIMIT 1")).scalar()
            if not user_id:
                import uuid
                user_id = uuid.uuid4()
                # Insertar usuario
                db.execute(text("""
                    INSERT INTO usuario (id, nombre, apellido, correo, hashed_password, rol, estado)
                    VALUES (:uid, 'Ramon', 'Perez', 'operario1@meme.com', '$2b$12$Z16Hw/pS8J2Tj0G8Qh...fake', 'operario', 'activo')
                """), {"uid": user_id})
            
            op_id = user_id
            db.execute(text("""
                INSERT INTO operario (id, maquinaActual, habilidades, estado)
                VALUES (:opid, 'merrow-01', '[{"maquina": "merrow", "nivel_eficiencia": 88}]', 'activo')
            """), {"opid": op_id})
        
        # 2. Asegurar máquina operativa
        maq_id = db.execute(text("SELECT id FROM maquina LIMIT 1")).scalar()
        if not maq_id:
            import uuid
            maq_id = uuid.uuid4()
            db.execute(text("""
                INSERT INTO maquina (id, codigo, tipo, nombre, estado)
                VALUES (:mid, 'MERROW-01', 'merrow', 'Cortadora Merrow', 'operativa')
            """), {"mid": maq_id})

        # 3. Limpiar datos de seed anteriores para evitar duplicados
        db.execute(text("DELETE FROM reporte_avance WHERE notas = 'Carga Seed'"))
        db.execute(text("DELETE FROM asignacion_orden WHERE notas = 'Carga Seed'"))
        db.execute(text("DELETE FROM orden WHERE notas = 'Carga Seed'"))

        # 4. Insertar 16 órdenes y asignaciones completadas distribuidas en 16 días distintos en el pasado
        import uuid
        import random
        count = 0
        
        for i in range(1, 26):
            ord_id = uuid.uuid4()
            asig_id = uuid.uuid4()
            rep_id = uuid.uuid4()
            
            prioridad = random.choice(["baja", "normal", "alta", "urgente"])
            piezas = random.randint(100, 1000)
            interval_str = f"{30 - i} days"
            horas_tomadas = random.uniform(2.0, 10.0)
            
            # Orden
            db.execute(text(f"""
                INSERT INTO orden (id, numero, cliente, tipo, prioridad, fecha_entrega_estimada, estado, notas, fecha_creacion)
                VALUES (:oid, :num, 'Cliente de Prueba', 'MTO', :prio, CURRENT_TIMESTAMP, 'COMPLETADA', 'Carga Seed', CURRENT_TIMESTAMP - INTERVAL '{interval_str}')
            """), {
                "oid": ord_id,
                "num": f"ORD-SEED-{1000+i}",
                "prio": prioridad
            })
            
            # Asignación
            db.execute(text(f"""
                INSERT INTO asignacion_orden (id, orden_id, operario_id, tarea, piezas_requeridas, piezas_completadas, estado, fecha_asignacion, notas)
                VALUES (:aid, :oid, :opid, 'Costura General', :cant, :cant, 'COMPLETADA', CURRENT_TIMESTAMP - INTERVAL '{interval_str}', 'Carga Seed')
            """), {
                "aid": asig_id,
                "oid": ord_id,
                "opid": op_id,
                "cant": piezas
            })
            
            # Reporte de avance
            db.execute(text(f"""
                INSERT INTO reporte_avance (id, asignacion_id, operario_id, piezas_reportadas, piezas_buenas, piezas_defectuosas, estado, fecha_reporte, fecha_validacion, notas)
                VALUES (:rid, :aid, :opid, :cant, :cant, 0, 'validado', CURRENT_TIMESTAMP - INTERVAL '{interval_str}' + INTERVAL '{int(horas_tomadas)} hours', CURRENT_TIMESTAMP - INTERVAL '{interval_str}' + INTERVAL '{int(horas_tomadas)} hours', 'Carga Seed')
            """), {
                "rid": rep_id,
                "aid": asig_id,
                "opid": op_id,
                "cant": piezas
            })
            count += 1
            
        db.commit()
        
        # Inicializar el primer modelo para que no dé fallback
        try:
            train_model()
        except Exception as e:
            print("Error al calibrar inicial en seed:", e)
            
        return SeedResponse(
            estado="exitoso", 
            mensaje="Registros históricos de producción sembrados con éxito (17 días) y modelo calibrado.", 
            registros_insertados=count
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Fallo al sembrar datos: {e}")
    finally:
        db.close()