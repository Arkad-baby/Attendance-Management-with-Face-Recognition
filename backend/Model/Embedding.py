# load libraries
from huggingface_hub import hf_hub_download
import os
import numpy as np
from PIL import Image
import face_recognition
from ultralytics import YOLO

class FaceDataEmbed:
    def __init__(self,dataset_path,model,db):
        self.dataset_path=dataset_path
        self.model=model
        self.db=db
        self.embeddings={}
        
    def imageEmbedding(self):
        for person_name in os.listdir(self.dataset_path):
            person_folder= os.path.join(self.dataset_path,person_name) 
            self.embeddings[person_name]=[]

            for img_name in os.listdir(person_folder):
                img_path=os.path.join(person_folder,img_name)
                self._imgEncoding(img_path,person_name)
    
    def _imgEncoding(self,img_path,person_name): 
            #Detect Face using YOLO
        results=self.model(img_path)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]   #get the co-ords of the face
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                img=Image.open(img_path).convert("RGB") #face_recognition use RGB
                face_crop=img.crop((x1,y1,x2,y2))
                face_array=np.array(face_crop)
                    
                #Get the encodings
                encodings=face_recognition.face_encodings(face_array)
                if encodings:
                    self.embeddings[person_name].append(encodings[0])
               


    def testImage(self,img):
        # img = face_recognition.load_image_file(img_path)

        results = self.model(img)
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

        self._compareEmbeddings(new_faces_embeddings)
        
        
    def _compareEmbeddings(self,new_faces_embeddings):
        from scipy.spatial.distance import euclidean
        response=db.readUserData(columns=["Username","Embeddings"])       
        for new_embedding in new_faces_embeddings:
            match_found=False
            for row in response.data:
                for emb in row["Embeddings"]:
                    distance=euclidean(emb,new_embedding)
                    if distance <0.4:
                        print(f"Attandence marked:{row["Username"]}")
                        print(row["Username"],distance)
                        match_found=True
                        break
                    
                if match_found:
                    break
            
            if not match_found:
                print("unknown Face")   


class DBUpload:
    def __init__(self,supabase):
        self.supabase=supabase
        
    def insertEmbedding(self,embeddings):
        for name, embedding in embeddings.items():
            embeddingList = []
            for emb in embedding:
                embeddingList.append(emb.tolist())
            data = {"Username": name, "Embeddings": embeddingList}

            response = self.supabase.table("Users").insert(data).execute()
            if response:
                print("Data inserted to DB successfully!")
                print(response)
                
    def readUserData(self,columns:list):
        columns=",".join(columns)
        response=self.supabase.table("Users").select(columns).execute()
        return response
            
            
from SupabaseClient import supabase

if __name__=="__main__":
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    # load model
    model = YOLO(model_path)
    dataset_path="DatasetOnce"
    
    db=DBUpload(supabase)
    
    faceDataEmbedding=FaceDataEmbed(dataset_path,model,db)
    # faceDataEmbedding.imageEmbedding()
    
    # db.insertEmbedding(faceDataEmbedding.embeddings)
    
    img_path="test/Obama.jpg"
    img=face_recognition.load_image_file(img_path)
    faceDataEmbedding.testImage(img)