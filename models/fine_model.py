import hashlib

MODEL_PATH = r"C:\FYPPP2\toxicbaddie\models\toxic_model.pkl"

def get_sha256(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        return hashlib.sha256(file_bytes).hexdigest()

print("NEW SHA256:", get_sha256(MODEL_PATH))