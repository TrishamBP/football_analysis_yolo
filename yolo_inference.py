from ultralytics import YOLO

model = YOLO("yolov8x")

results = model.predict("data/test.mp4",save=True)

print(results[0])

print("--------------------")

for box in results[0].boxes:
    print(box)