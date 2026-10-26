import os
from pathlib import Path
from ultralytics import YOLO

modelos = list(Path('runs/detect/DeltaV_Vision').rglob('best.pt'))

print(f"Total de modelos encontrados: {len(modelos)}")

for caminho in modelos:
    if os.path.exists(caminho):
        print(f"\n==========================================")
        print(f"EXPORTANDO MODELO: {caminho}")
        print(f"==========================================")
        
        pasta_pesos = caminho.parent
        
        for imgsz in [320, 512]:
            print(f"\n--> Exportando para resolução {imgsz}x{imgsz}...")
            
            # 1. FP32
            m1 = YOLO(str(caminho))
            out1 = m1.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
            os.rename(out1, pasta_pesos / f"best_fp32_{imgsz}.onnx")
            
            # 2. FP16
            m2 = YOLO(str(caminho))
            out2 = m2.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, half=True)
            os.rename(out2, pasta_pesos / f"best_fp16_{imgsz}.onnx")
            
            # 3. INT8
            m3 = YOLO(str(caminho))
            out3 = m3.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, int8=True, data="dataset_para_treino/data.yaml")
            os.rename(out3, pasta_pesos / f"best_int8_{imgsz}.onnx")

print("\n--- TODAS AS VARIANTES EXPORTADAS COM SUCESSO! ---")
