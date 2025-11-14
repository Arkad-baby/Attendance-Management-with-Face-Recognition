# load libraries
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO
import face_recognition
import numpy as np

# download model
model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
)

# load model
model = YOLO(model_path)

img_path = "virat.jpg"
img = face_recognition.load_image_file(img_path)

results = model(img)
new_faces_embeddings = []

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        top, right, bottom, left = y1, x2, y2, x1
        location = [(top, right, bottom, left)]
        encodings = face_recognition.face_encodings(img, known_face_locations=location)
        if encodings:
            new_faces_embeddings.append(encodings[0])




# Compare
from scipy.spatial.distance import euclidean

embedding_all = np.load("Embedding_file.npy", allow_pickle=True).item()

for new_embedding in new_faces_embeddings:
    match_found=False
    for name,emb_list in embedding_all.items():
        for emb in emb_list:
            distance=euclidean(emb,new_embedding)
            if distance <0.5:
                print(f"Attandence marked:{name}")
                print(name,distance)
                match_found=True
                break
            
        if match_found:
            break
    
    if not match_found:
        print("unknown Face")   