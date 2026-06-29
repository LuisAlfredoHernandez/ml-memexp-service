import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text
from core.config import settings
from core.database import SessionLocal

class DeliveryTimePredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carga el modelo en memoria al iniciar el servidor."""
        try:
            if os.path.exists(settings.MODEL_PATH):
                self.model = joblib.load(settings.MODEL_PATH)
                print(f"✅ Modelo cargado correctamente desde: {settings.MODEL_PATH}")
            else:
                print(f"⚠️ Advertencia: No existe el modelo en {settings.MODEL_PATH}. Usando fallback matemático.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}. Usando fallback matemático.")

    def predict(self, cantidad_piezas: int, prioridad_alta: bool, lineas_produccion: int) -> tuple[float, float]:
        """Predice el tiempo de entrega en horas (RF12)"""
        if self.model is None:
            # Fallback matemático realista
            factor = 0.35 if prioridad_alta else 0.45
            estimacion = (cantidad_piezas * factor) / lineas_produccion
            estimacion = max(1.0, estimacion)
            return float(round(estimacion, 2)), float(round(estimacion * 0.08, 2))

        try:
            df_entrada = pd.DataFrame([{
                "cantidad_piezas": cantidad_piezas,
                "prioridad_alta": 1 if prioridad_alta else 0,
                "lineas_produccion": lineas_produccion
            }])
            estimacion = self.model.predict(df_entrada)[0]
            margen_error = estimacion * 0.08
            return float(round(estimacion, 2)), float(round(margen_error, 2))
        except Exception as e:
            print(f"Error en predicción del modelo: {e}")
            # Fallback en caso de error de forma
            factor = 0.45
            estimacion = (cantidad_piezas * factor) / lineas_produccion
            return float(round(estimacion, 2)), 2.0

    def get_projections(self) -> list[dict]:
        """Genera proyecciones de nivel de producción diaria (RF13)"""
        db = SessionLocal()
        try:
            # Consultar los últimos 7 días reales de producción diaria validada
            query = text("""
                SELECT DATE(fecha_reporte) as dia, SUM(piezas_buenas) as total_piezas
                FROM reporte_avance
                WHERE estado = 'validado' AND fecha_reporte >= CURRENT_DATE - INTERVAL '10 days'
                GROUP BY DATE(fecha_reporte)
                ORDER BY dia ASC
            """)
            result = db.execute(query).fetchall()
            
            proyecciones = []
            dia_index = 1
            acumulado_real = 0
            
            for row in result:
                fecha_str = row[0].strftime("%d/%m")
                real_pzs = int(row[1])
                acumulado_real += real_pzs
                proyecciones.append({
                    "d": fecha_str,
                    "meta": dia_index * 30,
                    "real": acumulado_real,
                    "pred": None
                })
                dia_index += 1

            # Si no hay datos suficientes en base de datos, usamos un mock dinámico realista
            if len(proyecciones) == 0:
                mock_data = [
                    {"d": "Lun", "meta": 30, "real": 26, "pred": None},
                    {"d": "Mar", "meta": 60, "real": 58, "pred": None},
                    {"d": "Mie", "meta": 90, "real": 82, "pred": None},
                    {"d": "Jue", "meta": 120, "real": 114, "pred": None},
                    {"d": "Vie", "meta": 150, "real": 143, "pred": None},
                    {"d": "Sab", "meta": 180, "real": 169, "pred": None},
                    {"d": "Dom", "meta": 210, "real": 204, "pred": 204},
                    {"d": "Lun+", "meta": 240, "real": None, "pred": 231},
                    {"d": "Mar+", "meta": 270, "real": None, "pred": 259},
                    {"d": "Mie+", "meta": 300, "real": None, "pred": 288},
                ]
                return mock_data

            # Predecir los siguientes 3 días utilizando el promedio del avance actual
            promedio_diario = acumulado_real / len(proyecciones) if len(proyecciones) > 0 else 25
            ultimo_acumulado = acumulado_real
            
            # Ajustar la predicción basándose en si hay máquinas averiadas
            averiadas_count = db.execute(text("SELECT COUNT(*) FROM maquina WHERE estado::text != 'operativa'")).scalar() or 0
            factor_reduccion = max(0.7, 1.0 - (averiadas_count * 0.1))
            
            proyecciones[-1]["pred"] = acumulado_real # El último día real coincide con el inicio de predicción
            
            for i in range(1, 4):
                proyecciones.append({
                    "d": f"+{i}d",
                    "meta": (len(proyecciones)) * 30,
                    "real": None,
                    "pred": int(ultimo_acumulado + (promedio_diario * factor_reduccion * i))
                })

            return proyecciones
        finally:
            db.close()

    def detect_bottlenecks(self) -> dict:
        """Identifica cuellos de botella y recomienda balanceo de línea (RF15)"""
        db = SessionLocal()
        try:
            # 1. Consultar estado y carga de las máquinas
            # Máquinas con reportes de avería activos o en asignaciones activas
            query_maquinas = text("""
                SELECT m.codigo, m.tipo, m.estado,
                       COALESCE((
                           SELECT SUM(ao.piezas_requeridas - ao.piezas_completadas)
                           FROM asignacion_orden ao
                           JOIN operario op ON ao.operario_id = op.id
                           WHERE op."maquinaActual"::text = m.codigo AND ao.estado = 'en_proceso'
                       ), 0) as carga_pendiente
                FROM maquina m
            """)
            maquinas_db = db.execute(query_maquinas).fetchall()
            
            cuellos = []
            maquinas_saturadas = []
            
            for m in maquinas_db:
                codigo, tipo, estado, carga = m
                # Simular nivel de saturación basado en carga de piezas pendientes
                carga = int(carga)
                if estado != "operativa":
                    saturacion = 0
                    nivel = "advertencia" if estado == "mantenimiento" else "critica"
                    msg = f"Máquina {codigo} fuera de servicio ({estado})."
                else:
                    saturacion = min(98, max(20, int(carga * 0.2)))
                    if saturacion > 80:
                        nivel = "critica"
                        msg = f"Saturación crítica en {codigo} — cuello de botella en {tipo}."
                        maquinas_saturadas.append((codigo, tipo))
                    elif saturacion > 60:
                        nivel = "advertencia"
                        msg = f"Carga elevada en {codigo}. Redistribuir operarios."
                    else:
                        nivel = "info"
                        msg = f"Capacidad ociosa en {codigo}. Puede absorber pedidos."
                
                cuellos.append({
                    "maquina": codigo,
                    "nivel": nivel,
                    "sat": saturacion,
                    "impacto": round(carga * 0.05, 1) if saturacion > 60 else 0.0,
                    "msg": msg
                })

            # Si no hay máquinas registradas, proveemos mock
            if len(cuellos) == 0:
                cuellos = [
                    { "maquina": "MERROW-01", "nivel": "critica", "sat": 94, "impacto": 3.5, "msg": "Saturación crítica — riesgo de parada en 4 hrs." },
                    { "maquina": "MERROW-03", "nivel": "advertencia", "sat": 78, "impacto": 1.2, "msg": "Carga elevada. Redistribuir operarios." },
                    { "maquina": "DTF-01", "nivel": "info", "sat": 35, "impacto": 0, "msg": "Capacidad ociosa. Puede absorber estampado pendiente." }
                ]
                maquinas_saturadas = [("MERROW-01", "merrow")]

            # 2. Encontrar operarios disponibles para balancear
            # Buscamos operarios asignados a máquinas subutilizadas pero con habilidades en las máquinas saturadas
            recomendaciones = []
            rec_id = 1
            
            if maquinas_saturadas:
                for sat_cod, sat_tipo in maquinas_saturadas:
                    # Encontrar operarios que no estén en la máquina saturada pero tengan habilidad para ella
                    query_operarios = text("""
                        SELECT o.id, o.nombre, o.apellido, o."maquinaActual", o.habilidades
                        FROM operario o
                        WHERE o."maquinaActual" != :sat_cod
                    """)
                    ops = db.execute(query_operarios, {"sat_cod": sat_cod}).fetchall()
                    
                    for op in ops:
                        op_id, nombre, apellido, maq_actual, habs = op
                        # habs es una lista de diccionarios/objetos con {"maquina": "merrow", "nivel_eficiencia": 88}
                        import json
                        try:
                            # A veces viene como string de json o como dict directo
                            habilidades = json.loads(habs) if isinstance(habs, str) else habs
                        except:
                            habilidades = []
                            
                        habilidad_destino = next((h for h in habilidades if h.get("maquina") == sat_tipo), None)
                        
                        if habilidad_destino and len(recomendaciones) < 2:
                            eficiencia = habilidad_destino.get("nivel_eficiencia", 80)
                            recomendaciones.append({
                                "id": f"r{rec_id}",
                                "empleado": f"{nombre} {apellido}",
                                "origen": maq_actual,
                                "destino": sat_cod,
                                "ganancia": round((eficiencia / 100.0) * 3.0, 1),
                                "prioridad": "alta" if eficiencia > 85 else "media",
                                "justificacion": f"Subutilizado en {maq_actual}. Su alta eficiencia en {sat_tipo} ({eficiencia}%) resolverá el cuello de botella."
                            })
                            rec_id += 1

            # Si no hay operarios en DB calificados para balancear, devolvemos fallback
            if len(recomendaciones) == 0:
                recomendaciones = [
                    { "id": "r1", "empleado": "Josué Reyes", "origen": "COVER-02", "destino": "MERROW-01", "ganancia": 2.5, "prioridad": "alta", "justificacion": "Subutilizado en COVER-02 (58%). Moverlo reducirá saturación crítica." },
                    { "id": "r2", "empleado": "Carmen Méndez", "origen": "COVER-01", "destino": "MERROW-03", "ganancia": 1.8, "prioridad": "media", "justificacion": "Alta eficiencia en Merrow (88%). Optimizaría salida de joggers." }
                ]

            return {
                "cuellos": cuellos,
                "recomendaciones": recomendaciones
            }
        finally:
            db.close()

    def simulate_mts_impact(self, cantidad_piezas: int) -> list[dict]:
        """Simula cómo afectará añadir una orden MTS de N piezas a los pedidos MTO vigentes (RF16)"""
        db = SessionLocal()
        try:
            # Buscar órdenes MTO en proceso o pendientes
            query = text("""
                SELECT id, numero, prioridad, fecha_entrega_estimada
                FROM orden
                WHERE tipo = 'MTO' AND estado::text != 'completada'
                ORDER BY fecha_entrega_estimada ASC
            """)
            orders = db.execute(query).fetchall()
            
            simulacion = []
            
            # Si no hay órdenes vigentes, retornamos mock descriptivo
            if len(orders) == 0:
                return [
                    { "orden": "ORD-2026-0042", "antes": "17 Mar", "despues": "17 Mar", "impacto": "Sin impacto", "color": "#34d399" },
                    { "orden": "ORD-2026-0043", "antes": "22 Mar", "despues": "24 Mar", "impacto": "+2 días", "color": "#fbbf24" }
                ]

            # Calcular el impacto: cada 200 piezas de MTS retrasan las órdenes MTO en 1 día laboral
            dias_retraso = int(np.ceil(cantidad_piezas / 200.0))
            
            for o in orders:
                oid, numero, prio, fecha_entrega = o
                antes_dt = pd.to_datetime(fecha_entrega) if fecha_entrega else pd.Timestamp.now() + pd.Timedelta(days=5)
                
                # Pedidos urgentes MTO no sufren retraso (prioridad sobre MTS - RF15)
                retraso_aplicado = 0 if prio in ["urgente", "alta"] else dias_retraso
                despues_dt = antes_dt + pd.Timedelta(days=retraso_aplicado)
                
                impacto_str = "Sin impacto" if retraso_aplicado == 0 else f"+{retraso_aplicado} días"
                color = "#34d399" if retraso_aplicado == 0 else ("#f87171" if retraso_aplicado > 3 else "#fbbf24")
                
                simulacion.append({
                    "orden": numero,
                    "antes": antes_dt.strftime("%d %b"),
                    "despues": despues_dt.strftime("%d %b"),
                    "impacto": impacto_str,
                    "color": color
                })

            return simulacion
        finally:
            db.close()

# Instanciar singleton
predictor = DeliveryTimePredictor()