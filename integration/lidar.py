def lidar_process(shared_state):
    """
    Hilo LIDAR: detección inteligente con buffer + agrupación + fusión (igual al script standalone),
    pero en vez de imprimir, publica objetos en shared_state.update_lidar_objects().
    """

    print("[Hilo - LIDAR] Iniciando LIDAR (modo smart)...")

    from pyrplidar import PyRPlidar
    import time
    import math

    # --- Helpers (copiados del script que funciona) ---
    def angle_difference(angle1, angle2):
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    def normalize_angle(angle):
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle

    def are_objects_same(obj1, obj2):
        dist_diff = abs(obj1["min_distance"] - obj2["min_distance"])
        angle_diff = angle_difference(obj1["avg_angle"], obj2["avg_angle"])
        return (dist_diff <= OBJECT_DISTANCE_THRESHOLD and angle_diff <= OBJECT_ANGLE_THRESHOLD)

    def analyze_object(group):
        if not group:
            return None

        angles = [p[0] for p in group]
        distances = [p[1] for p in group]

        # promedio circular del ángulo
        x_sum = 0.0
        y_sum = 0.0
        for ang in angles:
            rad = math.radians(ang)
            x_sum += math.cos(rad)
            y_sum += math.sin(rad)

        x_avg = x_sum / len(angles)
        y_avg = y_sum / len(angles)
        avg_rad = math.atan2(y_avg, x_avg)
        avg_angle = normalize_angle(math.degrees(avg_rad))

        min_distance = min(distances)
        avg_distance = sum(distances) / len(distances)

        if len(angles) > 1:
            adjusted_angles = []
            for a in angles:
                if abs(angle_difference(a, avg_angle)) > 180:
                    a = a + 360 if a < avg_angle else a - 360
                adjusted_angles.append(a)
            angle_span = max(adjusted_angles) - min(adjusted_angles)
        else:
            angle_span = 0.0

        if angle_span > 0.1 and avg_distance > 0:
            width_est = 2 * avg_distance * math.tan(math.radians(angle_span / 2))
        else:
            width_est = 0.0

        if 0 <= avg_angle <= 60:
            position = "FRONTAL DERECHO"
        elif 300 <= avg_angle <= 360:
            position = "FRONTAL IZQUIERDO"
        elif 0 <= avg_angle <= 30 or 330 <= avg_angle <= 360:
            position = "CENTRO FRONTAL"
        else:
            position = "FRONTAL"

        return {
            "avg_angle": avg_angle,
            "min_distance": min_distance,
            "avg_distance": avg_distance,
            "angle_span": angle_span,
            "width_est": width_est,
            "point_count": len(group),
            "position": position,
            "timestamp": time.time(),
        }

    def merge_object_groups(groups):
        if not groups:
            return []

        merged_objects = []
        for group in groups:
            obj_info = analyze_object(group)
            if obj_info is None:
                continue

            merged = False
            for i, existing in enumerate(merged_objects):
                if are_objects_same(obj_info, existing["info"]):
                    merged_objects[i]["points"].extend(group)
                    merged_objects[i]["info"] = analyze_object(merged_objects[i]["points"])
                    merged = True
                    break

            if not merged:
                merged_objects.append({"points": list(group), "info": obj_info})

        return merged_objects

    def group_points_by_distance_and_angle(points):
        if len(points) < MIN_POINTS_PER_OBJECT:
            return []

        sorted_points = sorted(points, key=lambda p: p[0])

        groups = []
        current_group = []

        for angle, distance in sorted_points:
            if not current_group:
                current_group.append((angle, distance))
                continue

            last_angle, last_distance = current_group[-1]
            a_diff = angle_difference(angle, last_angle)
            d_diff = abs(distance - last_distance)

            if a_diff <= ANGLE_THRESHOLD and d_diff <= DISTANCE_THRESHOLD:
                current_group.append((angle, distance))
            else:
                if len(current_group) >= MIN_POINTS_PER_OBJECT:
                    groups.append(current_group)
                current_group = [(angle, distance)]

        if len(current_group) >= MIN_POINTS_PER_OBJECT:
            groups.append(current_group)

        return groups

    def should_alert_for_object(obj_info, last_alerts):
        now = time.time()
        for alert_time, alert_obj in last_alerts:
            if (now - alert_time) < MIN_TIME_BETWEEN_ALERTS:
                if are_objects_same(obj_info, alert_obj):
                    return False
        return True

    # --- Parámetros: preferimos CONFIG, pero caemos a defaults del script ---
    FRONT_ANGLE_MIN1 = float(CONFIG["lidar"].get("front_angle_min1", 0))
    FRONT_ANGLE_MAX1 = float(CONFIG["lidar"].get("front_angle_max1", 60))
    FRONT_ANGLE_MIN2 = float(CONFIG["lidar"].get("front_angle_min2", 300))
    FRONT_ANGLE_MAX2 = float(CONFIG["lidar"].get("front_angle_max2", 360))

    ALERT_DISTANCE = float(CONFIG["lidar"].get("alert_distance", 1000))  # mm

    DISTANCE_THRESHOLD = float(CONFIG["lidar"].get("distance_threshold", 50))  # mm
    ANGLE_THRESHOLD = float(CONFIG["lidar"].get("angle_gap_threshold", 5))     # grados

    MIN_POINTS_PER_OBJECT = 3
    BUFFER_SIZE = int(CONFIG["lidar"].get("buffer_size", 150))

    OBJECT_DISTANCE_THRESHOLD = float(CONFIG["lidar"].get("object_distance_threshold", 100))  # mm
    OBJECT_ANGLE_THRESHOLD = float(CONFIG["lidar"].get("object_angle_threshold", 15))         # grados
    MIN_TIME_BETWEEN_ALERTS = float(CONFIG["lidar"].get("min_time_between_alerts", 2.0))      # s

    PROCESS_EVERY_N_POINTS = int(CONFIG["lidar"].get("process_every_n_points", 40))

    # --- Conexión LIDAR ---
    lidar = PyRPlidar()

    try:
        lidar.connect(
            port=CONFIG["lidar"]["port"],
            baudrate=int(CONFIG["lidar"]["baudrate"]),  # IMPORTANTE: en tu caso real debe ser 469800
            timeout=3,
        )
        lidar.set_motor_pwm(int(CONFIG["lidar"]["motor_pwm"]))
        time.sleep(2)

        scan_generator = lidar.force_scan()

        point_buffer = []
        alert_history = []
        total_points = 0

        while True:
            # CLAVE: igual que el script que funciona (scan_generator())
            for scan in scan_generator():
                total_points += 1

                raw_angle = scan.angle
                distance = scan.distance

                angle = normalize_angle(raw_angle)

                in_frontal = (
                    (FRONT_ANGLE_MIN1 <= angle <= FRONT_ANGLE_MAX1)
                    or (FRONT_ANGLE_MIN2 <= angle <= FRONT_ANGLE_MAX2)
                )

                if in_frontal and distance > 0 and distance <= ALERT_DISTANCE * 1.2:
                    point_buffer.append((angle, distance))
                    if len(point_buffer) > BUFFER_SIZE:
                        point_buffer = point_buffer[-BUFFER_SIZE:]

                # Procesar cada N puntos (igual lógica)
                if total_points % PROCESS_EVERY_N_POINTS == 0:
                    now = time.time()

                    critical_points = [(a, d) for a, d in point_buffer if 0 < d <= ALERT_DISTANCE]

                    lidar_objects_out = {}

                    if len(critical_points) >= MIN_POINTS_PER_OBJECT * 2:
                        point_groups = group_points_by_distance_and_angle(critical_points)

                        if point_groups:
                            merged_objects = merge_object_groups(point_groups)

                            obj_idx = 0
                            for obj_data in merged_objects:
                                obj_info = obj_data["info"]
                                if obj_info is None:
                                    continue

                                # mismo throttling de alertas (si quieres usarlo)
                                if should_alert_for_object(obj_info, alert_history):
                                    lidar_objects_out[f"objeto_{obj_idx}"] = {
                                        "angle": float(obj_info["avg_angle"]),
                                        "distance": float(obj_info["min_distance"]),  # mm
                                        "position": obj_info["position"],
                                        "width_est": float(obj_info["width_est"]),
                                        "angle_span": float(obj_info["angle_span"]),
                                        "point_count": int(obj_info["point_count"]),
                                        "timestamp": float(obj_info["timestamp"]),
                                    }
                                    obj_idx += 1

                                    # registrar en historial para no spamear
                                    alert_history.append((now, obj_info))

                    # publicar objetos (vacío si no hay)
                    shared_state.update_lidar_objects(lidar_objects_out)

                    # limpiar historial viejo (últimos 10 s)
                    alert_history = [(t, obj) for t, obj in alert_history if (now - t) < 10.0]

                    # limpieza del buffer (igual que script)
                    if len(point_buffer) > int(BUFFER_SIZE * 0.8):
                        point_buffer = point_buffer[-int(BUFFER_SIZE * 0.7):]

    except Exception as e:
        print(f"[Hilo - LIDAR] ERROR critico: {e}")

    finally:
        print("[Hilo - LIDAR] Deteniendo LIDAR...")
        try:
            lidar.stop()
            lidar.set_motor_pwm(0)
            lidar.disconnect()
        except Exception:
            pass
