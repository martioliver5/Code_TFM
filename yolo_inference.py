from ultralytics import YOLO

#model = YOLO('models/best.pt') #Si no funciona provar yolov8x / yolov8l / yolov8m / yolov8s / yolov8n
model = YOLO('yolov8x')

results = model.predict('Prova_TFM.mp4', save = True)

print(results[0]) #Ens torna els resultats del primer frame
print('=====================================')
for box in results[0].boxes:
    print(box)
