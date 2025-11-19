# load libraries
from huggingface_hub import hf_hub_download
import os
import numpy as np
from PIL import Image
import face_recognition
from ultralytics import YOLO
import cv2
from scipy.spatial.distance import euclidean


class FaceDataEmbed:
    def __init__(self, dataset_path, model, db, User_data):
        self.dataset_path = dataset_path
        self.model = model
        self.User_data = User_data
        self.db = db
        self.embeddings = {}

    def imageEmbedding(self):
        for person_name in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person_name)
            self.embeddings[person_name] = []

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                self._imgEncoding(img_path, person_name)

    def _imgEncoding(self, img_path, person_name):
        # Detect Face using YOLO
        results = self.model(img_path)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # get the co-ords of the face
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                img = Image.open(img_path).convert("RGB")  # face_recognition use RGB
                face_crop = img.crop((x1, y1, x2, y2))
                face_array = np.array(face_crop)

                # Get the encodings
                encodings = face_recognition.face_encodings(face_array)
                if encodings:
                    self.embeddings[person_name].append(encodings[0])

    def runCamera(self):

        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            new_faces_embeddings = []
            boxes_xyxy = []
            results = self.model(frame)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # save for drawing later
                    boxes_xyxy.append((x1, y1, x2, y2))

                    top, right, bottom, left = y1, x2, y2, x1
                    location = [(top, right, bottom, left)]
                    encodings = face_recognition.face_encodings(
                        rgb_frame, known_face_locations=location
                    )
                    if encodings:
                        new_faces_embeddings.append(encodings[0])

            matches = self._compareEmbeddings(new_faces_embeddings)

            for ((x1, y1, x2, y2)), (best_match, best_distance) in zip(
                    boxes_xyxy, matches
                ):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    text = f"{best_match} ({best_distance:.2f})"

                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                    )
            cv2.imshow("Face Recognition", frame)
            # Press q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def testImage(self, img):
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
                encodings = face_recognition.face_encodings(
                    img, known_face_locations=location
                )
                if encodings:
                    new_faces_embeddings.append(encodings[0])

        matches = self._compareEmbeddings(new_faces_embeddings)
        return matches

    def _compareEmbeddings(self, new_faces_embeddings):

        matches = []
        for new_embedding in new_faces_embeddings:
            best_match = "Unknown"
            best_distance = 999

            for row in self.User_data:
                for emb in row["Embeddings"]:
                    distance = euclidean(emb, new_embedding)

                    if distance < best_distance and distance < 0.5:
                        best_match = row["Username"]
                        best_distance = distance

            matches.append((best_match, best_distance))
        return matches


class DBUpload:
    def __init__(self, supabase):
        self.supabase = supabase

    def insertEmbedding(self, embeddings):
        for name, embedding in embeddings.items():
            embeddingList = []
            for emb in embedding:
                embeddingList.append(emb.tolist())
            data = {"Username": name, "Embeddings": embeddingList}

            response = self.supabase.table("Users").insert(data).execute()
            if response:
                print("Data inserted to DB successfully!")
                print(response)

    def readUserData(self, columns: list):
        columns = ",".join(columns)
        response = self.supabase.table("Users").select(columns).execute()
        np.save("Users_Data.npy", response.data, allow_pickle=True)
        print("Dat retreived!")


from SupabaseClient import supabase

if __name__ == "__main__":
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
    )

    # load model
    model = YOLO(model_path)
    dataset_path = "DatasetOnce"

    db = DBUpload(supabase)
    db.readUserData(columns=["Username", "Embeddings"])

    # Load Users_Data
    User_data = np.load("Users_Data.npy", allow_pickle=True)
    User_data = User_data.tolist()

    faceDataEmbedding = FaceDataEmbed(dataset_path, model, db, User_data)
    # faceDataEmbedding.imageEmbedding()

    # db.insertEmbedding(faceDataEmbedding.embeddings)

    # #Checking 1 image
    # img_path = "test/Me.jpg"
    # img = face_recognition.load_image_file(img_path)
    # ans=faceDataEmbedding.testImage(img)
    # print(ans)

    # Running Camera
    faceDataEmbedding.runCamera()