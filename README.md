# 🚀 YOLO26 PoC - Explorador Web y Conteo con GPU (RTX 3060)

PoC interactiva para probar y exprimir las capacidades de **Ultralytics YOLO26** en entornos con aceleración por GPU (NVIDIA GeForce RTX 3060 12GB VRAM).

Incluye una interfaz web en navegador (**Gradio**) con soporte para detección general, segmentación por instancias, estimación de posturas (pose), **conteo de personas en tiempo real con seguimiento (Tracking de IDs únicos)** y exportación optimizada de vídeo (H.264).

---

## ✨ Características

- **Soporte completo de YOLO26:** Variantes `nano`, `small`, `medium`, `large` y `extra-large` (`n`, `s`, `m`, `l`, `x`), así como segmentación (`-seg`) y posturas (`-pose`).
- **👥 Modo Conteo de Personas:**
  - **En imagen:** Detección y conteo con cartel OSD superpuesto.
  - **En vídeo:** Conteo simultáneo de personas presentes en cuadro y **total acumulado de personas únicas** mediante algoritmos de tracking (`model.track` con `lapx` / ByteTrack).
- **📊 Barra de Progreso Integrada en Tiempo Real:** Avance frame a frame, porcentaje, FPS de procesamiento y tiempo restante estimado (ETA).
- **🎬 Codificación H.264 Directa:** Convierte automáticamente los vídeos con `ffmpeg` para reproducción fluida e inmediata en cualquier navegador.
- **💾 Guardado Automático y Descarga:** Cada resultado se persiste en `outputs/` con fecha/hora y modelo, y cuenta con botón de descarga con un clic en la interfaz web.
- **🐳 Dockerizado con GPU:** Incluye `Dockerfile` y `docker-compose.yml` listos para desplegar con NVIDIA Container Toolkit.

---

## 🚀 Despliegue con Docker (Recomendado)

### Requisitos previos
- Docker & Docker Compose
- Drivers NVIDIA y [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Iniciar con Docker Compose
```bash
# Construir e iniciar en segundo plano con acceso a GPU
docker compose up -d

# Ver logs en tiempo real
docker compose logs -f

# Detener el contenedor
docker compose down
```

Una vez iniciado, abre en tu navegador:
👉 **http://localhost:7860**

---

## 💻 Ejecución Local (sin Docker)

### Requisitos
- Python 3.10 o 3.11
- Tarjeta NVIDIA con controladores CUDA compatibles
- `ffmpeg` instalado en el sistema (`sudo apt install ffmpeg`)

### Paso a paso
1. **Crear y activar entorno virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

2. **Instalar PyTorch con aceleración CUDA:**
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
   ```

3. **Instalar requerimientos:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Opcional) Pre-descargar todos los modelos YOLO26:**
   ```bash
   python download_models.py
   ```
   *(Si omites este paso, cada modelo se descargará automáticamente la primera vez que lo selecciones en el combobox).*

5. **Iniciar la aplicación:**
   ```bash
   python app.py
   ```
   Abre **http://localhost:7860** en tu navegador.

---

## 📁 Estructura del Proyecto

```text
.
├── app.py                  # Servidor web Gradio con lógica de inferencia, OSD y tracking
├── download_models.py      # Script para verificar o descargar los 15 modelos YOLO26
├── Dockerfile              # Definición de la imagen Docker (PyTorch + CUDA + FFmpeg)
├── docker-compose.yml      # Configuración de Docker Compose con GPU passthrough
├── requirements.txt        # Dependencias de Python
├── samples/                # Imágenes de prueba para demostración rápida
│   ├── bus.jpg
│   └── zidane.jpg
└── outputs/                # Carpeta donde se guardan automáticamente las imágenes/vídeos procesados
```

---

## 📄 Licencia
Este repositorio es una prueba de concepto (PoC). Los pesos de YOLO26 se distribuyen bajo la licencia de Ultralytics (AGPL-3.0 / Enterprise).
