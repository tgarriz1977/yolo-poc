"""
YOLO26 Web UI - Inferencia interactiva en navegador con aceleración NVIDIA RTX 3060 (12 GB)
Estilos de alto contraste adaptados al modo oscuro para máxima legibilidad.
"""
import os
import time
import subprocess
import cv2
import numpy as np
import torch
import gradio as gr
from ultralytics import YOLO

OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/mnt/disk/yolo/outputs")
SAMPLES_DIR = os.getenv("SAMPLES_DIR", "/mnt/disk/yolo/samples")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Cache de modelos en memoria
LOADED_MODELS = {}

AVAILABLE_MODELS = [
    # Detección de objetos
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    # Segmentación por instancias
    "yolo26n-seg.pt",
    "yolo26s-seg.pt",
    "yolo26m-seg.pt",
    "yolo26l-seg.pt",
    "yolo26x-seg.pt",
    # Estimación de posturas (Pose)
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
]

def get_model(model_name: str):
    if model_name not in LOADED_MODELS:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_name)
        model.to(device)
        LOADED_MODELS[model_name] = model
    return LOADED_MODELS[model_name]

def get_gpu_stats():
    if not torch.cuda.is_available():
        return "⚠️ **GPU:** No disponible (ejecutando en CPU)"
    
    device_name = torch.cuda.get_device_name(0)
    allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    
    return (
        f"🎮 **GPU:** {device_name} (12 GB)\n\n"
        f"📊 **VRAM en uso:** {allocated_mb:.1f} MB | **Reservada:** {reserved_mb:.1f} MB | **Total disponible:** {total_mb:.0f} MB"
    )

def draw_osd_banner(img_bgr, text_lines):
    """Superpone un cartel semi-transparente estilizado en la esquina superior izquierda."""
    if not text_lines:
        return img_bgr
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    
    pad = 12
    line_h = 28
    box_w = min(max(int(w * 0.38), 340), w - 30)
    box_h = pad * 2 + len(text_lines) * line_h
    
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (25, 28, 36), -1)
    cv2.addWeighted(overlay, 0.8, img_bgr, 0.2, 0, img_bgr)
    cv2.rectangle(img_bgr, (15, 15), (15 + box_w, 15 + box_h), (94, 129, 172), 2)
    
    for i, line in enumerate(text_lines):
        y = 15 + pad + (i + 1) * line_h - 6
        cv2.putText(img_bgr, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (236, 239, 244), 2, cv2.LINE_AA)
    return img_bgr

def predict_image(image, model_name, conf_threshold, iou_threshold, imgsz, filter_people):
    if image is None:
        status_html = '''
        <div style="background-color: #2a2111; border: 1.5px solid #d97706; border-radius: 8px; padding: 12px 16px; color: #fbbf24; font-weight: bold;">
            ⚠️ Por favor sube o selecciona una imagen antes de ejecutar.
        </div>
        '''
        return None, status_html, "", None, get_gpu_stats()
    
    model = get_model(model_name)
    device = 0 if torch.cuda.is_available() else "cpu"
    classes_filter = [0] if filter_people else None
    
    t_start = time.perf_counter()
    results = model.predict(
        source=image,
        conf=float(conf_threshold),
        iou=float(iou_threshold),
        imgsz=int(imgsz),
        classes=classes_filter,
        device=device,
        verbose=False
    )
    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
    
    res = results[0]
    annotated_bgr = res.plot()
    
    names = model.names
    counts = {}
    num_boxes = len(res.boxes) if res.boxes is not None else 0
    person_count = 0
    
    if num_boxes > 0:
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for c in classes:
            cname = names.get(c, str(c))
            counts[cname] = counts.get(cname, 0) + 1
            if cname == 'person':
                person_count += 1
                
    if filter_people:
        draw_osd_banner(annotated_bgr, [f"Personas detectadas: {person_count}"])
    elif person_count > 0:
        draw_osd_banner(annotated_bgr, [f"Personas: {person_count} | Total obj: {num_boxes}"])
        
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Guardar en disco
    ts = time.strftime("%Y%m%d_%H%M%S")
    m_clean = model_name.replace(".pt", "")
    tag = "_personas" if filter_people else ""
    saved_img_path = os.path.join(OUTPUTS_DIR, f"img_{ts}_{m_clean}{tag}.jpg")
    cv2.imwrite(saved_img_path, annotated_bgr)
    
    summary = [
        f"⚡ **Inferencia:** {elapsed_ms:.2f} ms ({fps:.1f} FPS)",
        f"👥 **Personas detectadas:** **{person_count}**",
        f"🎯 **Objetos totales:** {num_boxes}",
        f"📁 **Guardado en disco:** `{saved_img_path}`"
    ]
    if counts and not filter_people:
        summary.append("\n**Detalle por clase:**")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            summary.append(f"- **{k}:** {v}")
            
    hora_actual = time.strftime("%H:%M:%S")
    status_html = f'''
    <div style="background-color: #0d2315; border: 1.5px solid #22c55e; border-radius: 8px; padding: 14px 18px; margin: 8px 0; font-family: system-ui, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 16px; font-weight: 800; color: #4ade80; letter-spacing: 0.3px;">✅ CONTEO COMPLETADO</span>
            <span style="font-size: 13px; color: #a7f3d0; background: #14351e; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{hora_actual}</span>
        </div>
        <div style="font-size: 15px; color: #f8fafc; margin-bottom: 6px; font-weight: 500;">
            👥 <b style="color: #4ade80; font-size: 17px;">Personas detectadas: {person_count}</b> 
            <span style="color: #cbd5e1;">| Total objetos: {num_boxes} (en {elapsed_ms:.1f} ms)</span>
        </div>
        <div style="font-size: 12px; color: #94a3b8; word-break: break-all;">
            💾 <span style="color: #cbd5e1;">Guardado en:</span> <code style="background: #020617; color: #38bdf8; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{saved_img_path}</code>
        </div>
    </div>
    '''
    
    gr.Info(f"👥 {person_count} persona(s) detectada(s) en {elapsed_ms:.1f} ms")
    return annotated_rgb, status_html, "\n".join(summary), saved_img_path, get_gpu_stats()

def predict_video(video_path, model_name, conf_threshold, iou_threshold, imgsz, filter_people, enable_tracking):
    if not video_path:
        status_html = '''
        <div style="background-color: #2a2111; border: 1.5px solid #d97706; border-radius: 8px; padding: 12px 16px; color: #fbbf24; font-weight: bold;">
            ⚠️ Por favor sube un archivo de vídeo antes de iniciar.
        </div>
        '''
        yield None, status_html, "", None, get_gpu_stats()
        return
        
    init_html = '''
    <div style="background: #18202f; border: 1.5px solid #3b82f6; border-radius: 8px; padding: 14px 18px; color: #93c5fd;">
        <div style="font-weight: 700; font-size: 15px;">⏳ Inicializando modelo con Tracking en RTX 3060...</div>
    </div>
    '''
    yield None, init_html, "Iniciando...", None, get_gpu_stats()
    
    model = get_model(model_name)
    device = 0 if torch.cuda.is_available() else "cpu"
    classes_filter = [0] if filter_people else None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    m_clean = model_name.replace(".pt", "")
    raw_output = os.path.join(OUTPUTS_DIR, f"temp_raw_{ts}.mp4")
    final_output = os.path.join(OUTPUTS_DIR, f"video_{ts}_{m_clean}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output, fourcc, fps_in, (width, height))
    
    t_start = time.perf_counter()
    frame_count = 0
    last_yield_time = 0
    
    unique_person_ids = set()
    current_person_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if enable_tracking:
            results = model.track(
                source=frame_rgb,
                persist=True,
                conf=float(conf_threshold),
                iou=float(iou_threshold),
                imgsz=int(imgsz),
                classes=classes_filter,
                device=device,
                verbose=False
            )
        else:
            results = model.predict(
                source=frame_rgb,
                conf=float(conf_threshold),
                iou=float(iou_threshold),
                imgsz=int(imgsz),
                classes=classes_filter,
                device=device,
                verbose=False
            )
            
        res = results[0]
        annotated_bgr = res.plot()
        
        names = model.names
        boxes = res.boxes
        current_person_count = 0
        
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.int().cpu().numpy().tolist() if (enable_tracking and boxes.id is not None) else None
            
            for i, c in enumerate(classes):
                if names.get(c) == 'person':
                    current_person_count += 1
                    if ids is not None and i < len(ids):
                        unique_person_ids.add(ids[i])
                        
        if filter_people:
            if enable_tracking:
                banner_lines = [
                    f"Personas en cuadro: {current_person_count}",
                    f"Personas unicas totales: {len(unique_person_ids)}"
                ]
            else:
                banner_lines = [f"Personas en cuadro: {current_person_count}"]
        else:
            if enable_tracking and len(unique_person_ids) > 0:
                banner_lines = [
                    f"Personas en cuadro: {current_person_count}",
                    f"Personas unicas: {len(unique_person_ids)}"
                ]
            else:
                banner_lines = [f"Objetos en cuadro: {len(boxes) if boxes is not None else 0}"]
                
        draw_osd_banner(annotated_bgr, banner_lines)
        out.write(annotated_bgr)
        frame_count += 1
        
        curr_time = time.perf_counter()
        if (curr_time - last_yield_time > 0.3) or (frame_count == total_frames):
            last_yield_time = curr_time
            elapsed_now = curr_time - t_start
            fps_now = frame_count / elapsed_now if elapsed_now > 0 else 0
            pct = int((frame_count / total_frames) * 100)
            remaining_frames = max(0, total_frames - frame_count)
            eta_sec = (remaining_frames / fps_now) if fps_now > 0 else 0
            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec >= 60 else f"{int(eta_sec)}s"
            elapsed_str = f"{int(elapsed_now // 60)}m {int(elapsed_now % 60):02d}s" if elapsed_now >= 60 else f"{int(elapsed_now)}s"
            
            tracking_info = f"| 👥 <b style='color: #4ade80;'>{len(unique_person_ids)}</b> personas unicas" if enable_tracking else ""
            
            progress_html = f'''
            <div style="background: #0f172a; border-radius: 10px; padding: 16px; border: 1.5px solid #334155; box-shadow: 0 4px 10px rgba(0,0,0,0.3); font-family: system-ui, sans-serif;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: 700; font-size: 16px; color: #38bdf8;">🚀 Frame {frame_count}/{total_frames} ({pct}%) {tracking_info}</span>
                    <span style="background: #1e293b; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; color: #4ade80;">⚡ {fps_now:.1f} FPS</span>
                </div>
                <div style="width: 100%; background-color: #334155; border-radius: 8px; overflow: hidden; height: 20px; margin-bottom: 10px;">
                    <div style="width: {pct}%; background: linear-gradient(90deg, #3b82f6, #38bdf8, #a855f7); height: 100%; transition: width 0.25s ease;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: #e2e8f0; font-weight: 500;">
                    <span>⏱️ Transcurrido: <b style="color: #ffffff;">{elapsed_str}</b></span>
                    <span>⏳ Restante estimado: <b style="color: #facc15;">~{eta_str}</b></span>
                </div>
            </div>
            '''
            yield None, progress_html, f"Procesando... Personas únicas acumuladas: {len(unique_person_ids)}", None, get_gpu_stats()
            
    cap.release()
    out.release()
    
    encoding_html = f'''
    <div style="background: #2a2111; border-radius: 8px; padding: 14px 18px; border: 1.5px solid #d97706; color: #fde047;">
        <div style="font-weight: 700; font-size: 15px;">🎬 {frame_count} frames procesados. Codificando vídeo H.264 para el navegador...</div>
    </div>
    '''
    yield None, encoding_html, "Optimizando con ffmpeg...", None, get_gpu_stats()
    
    cmd = [
        "ffmpeg", "-y", "-i", raw_output,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", final_output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_output):
        try:
            os.remove(raw_output)
        except Exception:
            pass
            
    total_time = time.perf_counter() - t_start
    avg_fps = frame_count / total_time if total_time > 0 else 0
    hora_actual = time.strftime("%H:%M:%S")
    
    tracking_result_str = f"👥 <b style='color: #4ade80; font-size: 17px;'>Total personas únicas detectadas en todo el vídeo: {len(unique_person_ids)}</b><br>" if enable_tracking else ""
    
    final_html = f'''
    <div style="background-color: #0d2315; border: 1.5px solid #22c55e; border-radius: 8px; padding: 16px 20px; font-family: system-ui, sans-serif; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 800; font-size: 17px; color: #4ade80;">🎉 ¡VÍDEO COMPLETADO Y GUARDADO!</span>
            <span style="font-size: 13px; color: #a7f3d0; background: #14351e; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{hora_actual}</span>
        </div>
        <div style="color: #f8fafc; font-size: 15px; margin-bottom: 8px; font-weight: 500;">
            {tracking_result_str}
            Total frames procesados: <b>{frame_count}</b> a <b>{avg_fps:.1f} FPS</b> (en <b>{total_time:.1f}s</b>).
        </div>
        <div style="font-size: 12px; color: #94a3b8; word-break: break-all;">
            💾 <span style="color: #cbd5e1;">Guardado en:</span> <code style="background: #020617; color: #38bdf8; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{final_output}</code>
        </div>
    </div>
    '''
    summary = (
        f"🎬 **Vídeo listo:** {frame_count} frames a {avg_fps:.1f} FPS en {total_time:.2f}s.\n\n"
        f"👥 **Personas únicas acumuladas (Tracking):** {len(unique_person_ids)}\n\n"
        f"📁 **Ruta en disco:** `{final_output}`"
    )
    gr.Info(f"🎉 ¡Conteo completado! {len(unique_person_ids)} persona(s) única(s) identificadas.")
    yield final_output, final_html, summary, final_output, get_gpu_stats()

# Lista de imágenes de prueba locales
EXAMPLES = []
for fname in ["bus.jpg", "zidane.jpg"]:
    fpath = os.path.join(SAMPLES_DIR, fname)
    if os.path.exists(fpath):
        EXAMPLES.append([fpath, "yolo26s.pt", 0.25, 0.45, 640, True])

# Interfaz Gradio
with gr.Blocks(title="YOLO26 - RTX 3060 Demo") as demo:
    gr.Markdown(
        """
        # 🚀 Ultralytics YOLO26 - Conteo y Detección en GPU
        Acelerado en tiempo real por tu **NVIDIA GeForce RTX 3060 (12 GB VRAM)**.
        Soporte para detección general y **Modo Conteo de Personas con Tracking de IDs únicos**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                label="Modelo YOLO26 (Descargados localmente)",
                choices=AVAILABLE_MODELS,
                value="yolo26s.pt",
                info="n: nano, s: small, m: medium, l: large, x: extra-large | -seg: segmentación | -pose: posturas"
            )
            
            with gr.Group():
                gr.Markdown("### 👥 Opciones de Conteo")
                filter_people_chk = gr.Checkbox(
                    label="Contar SOLO personas (Filtrar clase 'person')",
                    value=True,
                    info="Filtra la detección para enfocarse exclusivamente en seres humanos."
                )
                enable_tracking_chk = gr.Checkbox(
                    label="Seguimiento / Tracking de IDs únicos (solo vídeo)",
                    value=True,
                    info="Asigna un ID a cada persona para contar personas únicas acumuladas sin duplicados."
                )
            
            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Umbral de Confianza (Confidence)"
            )
            iou_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.45,
                step=0.05,
                label="Umbral de Solapamiento (IoU NMS)"
            )
            imgsz_radio = gr.Radio(
                choices=[320, 640, 1024, 1280],
                value=640,
                label="Resolución de Entrada (imgsz)"
            )
            gpu_info = gr.Markdown(value=get_gpu_stats)
            btn_refresh_gpu = gr.Button("🔄 Actualizar Telemetría GPU", size="sm")
            btn_refresh_gpu.click(fn=get_gpu_stats, outputs=[gpu_info])
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🖼️ Imagen"):
                    input_img = gr.Image(type="numpy", label="Subir Imagen o Seleccionar Ejemplo")
                    btn_predict_img = gr.Button("👥 Contar Personas / Detectar", variant="primary", size="lg")
                    status_img = gr.HTML(
                        '<div style="background-color: #1e293b; color: #94a3b8; border: 1.5px solid #334155; padding: 12px 16px; border-radius: 8px; font-weight: 500;">⏸️ Estado: En espera de imagen...</div>'
                    )
                    output_img = gr.Image(type="numpy", label="Resultado Anotado con Contador OSD")
                    download_img_file = gr.File(label="📥 Descargar Imagen Anotada (.jpg)")
                    output_info = gr.Markdown()
                    
                    if EXAMPLES:
                        gr.Examples(
                            examples=EXAMPLES,
                            inputs=[input_img, model_dropdown, conf_slider, iou_slider, imgsz_radio, filter_people_chk],
                            outputs=[output_img, status_img, output_info, download_img_file, gpu_info],
                            fn=predict_image,
                            cache_examples=False,
                            label="Ejemplos rápidos"
                        )
                    
                    btn_predict_img.click(
                        fn=predict_image,
                        inputs=[input_img, model_dropdown, conf_slider, iou_slider, imgsz_radio, filter_people_chk],
                        outputs=[output_img, status_img, output_info, download_img_file, gpu_info],
                        show_progress="minimal"
                    )

                with gr.TabItem("🎥 Cámara Web (Live)"):
                    webcam_img = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam en Directo")
                    webcam_output = gr.Image(type="numpy", label="Detección y Conteo en Tiempo Real")
                    webcam_stats = gr.Markdown()
                    
                    webcam_img.stream(
                        fn=predict_image,
                        inputs=[webcam_img, model_dropdown, conf_slider, iou_slider, imgsz_radio, filter_people_chk],
                        outputs=[webcam_output, gr.HTML(visible=False), webcam_stats, gr.File(visible=False), gpu_info]
                    )

                with gr.TabItem("🎬 Archivo de Vídeo"):
                    input_video = gr.Video(label="Subir Vídeo (mp4, avi, mkv, mov)")
                    btn_predict_vid = gr.Button("▶️ Iniciar Conteo y Detección en GPU", variant="primary", size="lg")
                    status_vid = gr.HTML(
                        '<div style="background-color: #1e293b; color: #94a3b8; border: 1.5px solid #334155; padding: 12px 16px; border-radius: 8px; font-weight: 500;">⏸️ Estado: En espera de vídeo...</div>'
                    )
                    output_video = gr.Video(label="Vídeo Procesado con Conteo Superpuesto")
                    download_video_file = gr.File(label="📥 Descargar Archivo de Vídeo Procesado (.mp4)")
                    output_vid_info = gr.Markdown()
                    
                    btn_predict_vid.click(
                        fn=predict_video,
                        inputs=[input_video, model_dropdown, conf_slider, iou_slider, imgsz_radio, filter_people_chk, enable_tracking_chk],
                        outputs=[output_video, status_vid, output_vid_info, download_video_file, gpu_info],
                        show_progress="minimal"
                    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
