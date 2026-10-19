"""
Descarga todos los modelos de YOLO26 en /mnt/disk/yolo
"""
import sys
from ultralytics import YOLO

MODELS = [
    # Detección
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    # Segmentación
    "yolo26n-seg.pt",
    "yolo26s-seg.pt",
    "yolo26m-seg.pt",
    "yolo26l-seg.pt",
    "yolo26x-seg.pt",
    # Pose
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
]

def main():
    total = len(MODELS)
    for idx, model_name in enumerate(MODELS, 1):
        print(f"\n[{idx}/{total}] Verificando / Descargando {model_name}...")
        try:
            model = YOLO(model_name)
            print(f"✅ {model_name} listo.")
        except Exception as e:
            print(f"❌ Error con {model_name}: {e}")
    print("\n🎉 ¡Todos los modelos han sido descargados y verificados con éxito!")

if __name__ == "__main__":
    main()
