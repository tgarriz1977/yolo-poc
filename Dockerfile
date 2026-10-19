# Imagen base oficial con PyTorch 2.6 y aceleración CUDA 12.4 lista
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Evitar prompts interactivos durante apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar ffmpeg y dependencias necesarias para procesamiento de vídeo y OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y muestras
COPY app.py .
COPY samples/ ./samples/

# Copiar todos los modelos .pt descargados para que la imagen sea 100% autónoma
COPY *.pt ./

# Configurar variables de entorno y directorios
ENV OUTPUTS_DIR=/app/outputs
ENV SAMPLES_DIR=/app/samples
ENV PORT=7860

# Crear directorio de salidas
RUN mkdir -p /app/outputs

EXPOSE 7860

CMD ["python", "app.py"]
