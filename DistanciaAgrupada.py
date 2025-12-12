from pyrplidar import PyRPLidar
import time
import math

# Configuracion del rango frontal y distancia de alerta
FRONT_ANGLE_MIN1 = 0      # Primer rango frontal: 0 a 60 grados
FRONT_ANGLE_MAX1 = 60
FRONT_ANGLE_MIN2 = 300    # Segundo rango frontal: 300 a 360 grados  
FRONT_ANGLE_MAX2 = 360
ALERT_DISTANCE = 1000     # Distancia de alerta en mm (1 metro)

# Parametros para agrupar objetos
DISTANCE_THRESHOLD = 50   # Maxima diferencia de distancia para considerar mismo objeto (mm)
ANGLE_GAP_THRESHOLD = 5   # Maximo salto de angulo para considerar mismo objeto (grados)

def simple_scan_with_alerts():
    """Escaneo simple con alertas de cercania en sector frontal"""
    lidar = PyRPLidar()
    lidar.connect(port="/dev/ttyUSB0", baudrate=469800, timeout=3)
    
    lidar.set_motor_pwm(500)
    time.sleep(2)
    
    scan_generator = lidar.force_scan()
    
    try:
        current_object = None  # Para agrupar puntos del mismo objeto
        object_points = []     # Lista de puntos del objeto actual
        
        for scan in scan_generator:
            angle = scan.angle
            distance = scan.distance
            
            # Normalizar angulo a 0-360
            angle_norm = angle % 360
            
            # Verificar si esta en sector frontal
            in_front_sector = ((0 <= angle_norm <= 60) or (300 <= angle_norm <= 360))
            
            # Verificar distancia critica
            is_critical_distance = (distance <= ALERT_DISTANCE)
            
            # Si es un punto de alerta en sector frontal
            if in_front_sector and is_critical_distance:
                
                if not current_object:
                    # Empezar nuevo objeto
                    current_object = {
                        'min_angle': angle_norm,
                        'max_angle': angle_norm,
                        'min_distance': distance,
                        'max_distance': distance,
                        'avg_distance': distance,
                        'count': 1
                    }
                    object_points = [(angle_norm, distance)]
                    
                else:
                    # Verificar si pertenece al mismo objeto
                    last_angle = object_points[-1][0]
                    last_distance = object_points[-1][1]
                    # Manejar el caso especial de cruce por 0/360 grados
                    if last_angle > 350 and angle_norm < 10:
                        # Ajustar angulo para continuidad
                        angle_for_check = angle_norm + 360
                    else:
                        angle_for_check = angle_norm
                    
                    if last_angle > 350 and angle_norm < 10:
                        last_angle_for_check = last_angle
                    else:
                        last_angle_for_check = last_angle
                    
                    angle_diff = abs(angle_for_check - last_angle_for_check)
                    distance_diff = abs(distance - last_distance)
                    
                    if angle_diff <= ANGLE_GAP_THRESHOLD and distance_diff <= DISTANCE_THRESHOLD:
                        # Es el mismo objeto
                        object_points.append((angle_norm, distance))
                        
                        # Actualizar estadisticas del objeto
                        current_object['min_angle'] = min(current_object['min_angle'], angle_norm)
                        current_object['max_angle'] = max(current_object['max_angle'], angle_norm)
                        current_object['min_distance'] = min(current_object['min_distance'], distance)
                        current_object['max_distance'] = max(current_object['max_distance'], distance)
                        
                        # Recalcular distancia promedio
                        total_distance = current_object['avg_distance'] * current_object['count']
                        current_object['count'] += 1
                        current_object['avg_distance'] = (total_distance + distance) / current_object['count']
                        
                    else:
                        # Finalizar objeto anterior y mostrar alerta
                        show_object_alert(current_object)
                        
                        # Empezar nuevo objeto
                        current_object = {
                            'min_angle': angle_norm,
                            'max_angle': angle_norm,
                            'min_distance': distance,
                            'max_distance': distance,
                            'avg_distance': distance,
                            'count': 1
                        }
                        object_points = [(angle_norm, distance)]
                        
                # No mostrar alertas individuales, solo mostramos informacion normal
                print(f"Angle: {angle_norm:.1f} grados, Distance: {distance:.1f} mm")
                
            else:
                # Si tenemos un objeto pendiente y cambiamos a sector no frontal
                if current_object:
                    show_object_alert(current_object)
                    current_object = None
                    object_points = []
                
                # Mostrar informacion normal
                print(f"Angle: {angle_norm:.1f} grados, Distance: {distance:.1f} mm")
                
    except KeyboardInterrupt:
        # Mostrar ultimo objeto si existe
        if current_object:
            show_object_alert(current_object)
        print("\n\nEscaneo interrumpido por el usuario")
        
    finally:
        print("\nDeteniendo LIDAR...")
        lidar.stop()
        lidar.set_motor_pwm(0)
        lidar.disconnect()
        print("LIDAR detenido correctamente")

def show_object_alert(obj):
    """Muestra alerta para un objeto agrupado"""
    # Calcular ancho del objeto en grados
    angle_span = obj['max_angle'] - obj['min_angle']
    
    # Ajustar para el caso especial de cruce por 0/360
    if obj['min_angle'] > 350 and obj['max_angle'] < 10:
        # El objeto cruza el punto 0/360
        angle_span = (360 - obj['min_angle']) + obj['max_angle']
    
    print("\n" + "!"*50)
    print("ALERTA DE OBJETO CERCANO!")
    print("!"*50)
    print(f"* Objeto detectado en sector frontal")
    print(f"* Rango de angulos: {obj['min_angle']:.1f} a {obj['max_angle']:.1f} grados")
    print(f"* Ancho aproximado: {angle_span:.1f} grados")
    print(f"* Distancia promedio: {obj['avg_distance']:.1f} mm")
    print(f"* Distancia minima: {obj['min_distance']:.1f} mm")
    print(f"* Distancia maxima: {obj['max_distance']:.1f} mm")
    print(f"* Puntos detectados: {obj['count']}")
    print(f"* Recomendacion: Reducir velocidad o cambiar direccion")
    print("-" * 50)

if __name__ == "__main__":
    simple_scan_with_alerts()
