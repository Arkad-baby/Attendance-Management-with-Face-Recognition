# load libraries
from huggingface_hub import hf_hub_download
import os
import numpy as np
from PIL import Image
import face_recognition
from ultralytics import YOLO

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

dataset_path="DatasetOnce"
embeddings={}

#['Anushka_Sharma', 'Barack_Obama', 'Bill_Gates', 'Dalai_Lama', 'Narendra_Modi']
for person_name in os.listdir(dataset_path):
    #Dataset\Barack_Obama
    person_folder= os.path.join(dataset_path,person_name) 
    embeddings[person_name]=[]

    for img_name in os.listdir(person_folder):
        img_path=os.path.join(person_folder,img_name)
        
        #Detect Face using YOLO
        results=model(img_path)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                img=Image.open(img_path).convert("RGB")
                face_crop=img.crop((x1,y1,x2,y2))
                face_array=np.array(face_crop)
                
                #Get the encodings
                encodings=face_recognition.face_encodings(face_array)
                if encodings:
                    embeddings[person_name].append(encodings[0])
                
# np.save("Embedding_file2.npy", embeddings)
print("Embeddings saved!")


#Insert to Database:
from SupabaseClient import supabase
import numpy as np


for name, embedding in embeddings.items():
    embeddingList = []
    for emb in embedding:
        embeddingList.append(emb.tolist())
    data = {"Username": name, "Embeddings": embeddingList}

    response = supabase.table("Users").insert(data).execute()
    print(response)
