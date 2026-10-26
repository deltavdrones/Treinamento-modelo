import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

base_dir = Path('runs/detect/DeltaV_Vision')
onnx_files = sorted(list(base_dir.rglob('*.onnx')))

print(f"Total de variantes ONNX encontradas: {len(onnx_files)}\n")

resultados = []

for path in onnx_files:
    modelo_nome = path.parent.parent.name.replace('detector_', '')
    arquivo = path.name
    tamanho_mb = os.path.getsize(path) / (1024 * 1024)
    
    # Identifica a resolução pelo nome do arquivo
    res = 320 if '320' in arquivo else 512
    
    print(f"Avaliando {modelo_nome} -> {arquivo} ({res}x{res})...")
    
    model = YOLO(str(path), task='detect')
    metrics = model.val(data="dataset_para_treino/data.yaml", split="val", imgsz=res, verbose=False)
    
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    latencia_ms = metrics.speed['inference']
    fps = 1000 / latencia_ms if latencia_ms > 0 else 0
    
    resultados.append({
        'Modelo': modelo_nome,
        'Arquivo': arquivo,
        'Resolução': f"{res}x{res}",
        'Tamanho (MB)': round(tamanho_mb, 2),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1_score, 4),
        'mAP50': round(map50, 4),
        'mAP50-95': round(map50_95, 4),
        'Latência (ms)': round(latencia_ms, 2),
        'FPS (PC)': round(fps, 1)
    })

df = pd.DataFrame(resultados)
df.to_csv('matriz_comparativa_completa.csv', index=False)

print("\n==================================================")
print("BENCHMARK COMPLETO CONCLUÍDO!")
print("==================================================")
print(df.to_string())