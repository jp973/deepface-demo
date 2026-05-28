
import os
from deepface import DeepFace
import cv2
import hashlib
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRIBUTORS_DIR = os.path.join(BASE_DIR, "assets", "contributors")
SELFIE_PATH = os.path.join(BASE_DIR, "assets", "selfie", "yash1_test.png")  #sharuhkan_test.jpg   yash1_test.png  deepika_test.png  radhika_Test.png

# Load contributor images and create embeddings with face_id for multiple faces
def build_face_database():
    database = {}  # {face_hash: {filenames: [list], embeddings: [list]}}
    print("🗄️  Building face database...")
    for file_name in os.listdir(CONTRIBUTORS_DIR):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(CONTRIBUTORS_DIR, file_name)
            try:
                # Use RetinaFace for better side face and occlusion detection
                representations = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False, detector_backend='retinaface')
                embeddings = [rep["embedding"] for rep in representations]
                print(f"  📸 Processing {file_name}: Detected {len(embeddings)} faces")
                for embedding in embeddings:
                    emb_hash = hashlib.md5(np.array(embedding).tobytes()).hexdigest()
                    if emb_hash not in database:
                        database[emb_hash] = {"filenames": [], "embeddings": []}
                    database[emb_hash]["filenames"].append(file_name)
                    database[emb_hash]["embeddings"].append(embedding)
                    print(f"    🔑 Created face_hash: {emb_hash[:10]}... for {file_name}")  # Shortened hash for readability
            except ValueError as e:
                print(f"  ❌ Skipped {file_name} due to error: {e}")
    print(f"✅ Database built successfully with {len(database)} unique face_hashes")
    return database

# Compare selfie with database and assign face_id for multiple faces
def recognize_selfie(database, selfie_path, threshold=0.4):
    matches = []
    print(f"🔍 Recognizing selfie: {os.path.basename(selfie_path)}...")
    selfie_representations = DeepFace.represent(selfie_path, model_name="Facenet", enforce_detection=False, detector_backend='retinaface')
    print(f"  👤 Detected {len(selfie_representations)} face(s) in selfie")

    for selfie_rep in selfie_representations:
        selfie_embedding = selfie_rep["embedding"]
        for face_hash, data in database.items():
            for db_embedding in data["embeddings"]:
                from numpy import dot
                from numpy.linalg import norm
                similarity = dot(selfie_embedding, db_embedding) / (norm(selfie_embedding) * norm(db_embedding))
                similarity_percent = similarity * 100
                if similarity_percent > (1 - threshold) * 100:
                    print(f"    🎯 Match found! hash: {face_hash[:10]}... (files: {data['filenames']}): Similarity = {similarity_percent:.2f}%")
                    matches.append({
                        "face_hash": face_hash,
                        "filenames": data["filenames"],
                        "Similarity": similarity_percent
                    })

    unique_matches = {}
    for match in matches:
        filename_key = tuple(sorted(match["filenames"]))
        if filename_key not in unique_matches or match["Similarity"] > unique_matches[filename_key]["Similarity"]:
            unique_matches[filename_key] = match
    return sorted(list(unique_matches.values()), key=lambda x: x["Similarity"], reverse=True)

def main():
    print("🚀 Starting DeepFace Demo...")
    print("\n" + "="*50)
    print("📂 INDEXING CONTRIBUTOR IMAGES")
    print("="*50)
    face_db = build_face_database()

    print("\n" + "="*50)
    print("🕵️  RUNNING SELFIE RECOGNITION")
    print("="*50)
    results = recognize_selfie(face_db, SELFIE_PATH)

    print("\n" + "="*50)
    print("📊 RESULT SUMMARY")
    print("="*50)
    if not results:
        print("  ❌ No matches found for the selfie.")
    else:
        for r in results:
            print(f"  ✅ Match: {', '.join(r['filenames'])} | Similarity: {r['Similarity']:.2f}%")

if __name__ == "__main__":
    main()