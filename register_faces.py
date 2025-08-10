import os, argparse, numpy as np
from insightface.app import FaceAnalysis
from utils import load_faiss, save_faiss, load_names, save_names, next_id, l2_normalize, EMBED_DIM
from PIL import Image
import cv2

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

def name_from_filename(fname: str) -> str:
    # remove extension and convert underscores to spaces, title case
    base = os.path.splitext(os.path.basename(fname))[0]
    name = base.replace('_', ' ').strip()
    return ' '.join([p.capitalize() for p in name.split()])

def get_embedding(app, img):
    faces = app.get(img)
    if not faces:
        return None
    # choose largest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    return np.array(emb, dtype='float32')

def register_all(app, images_dir=IMAGES_DIR):
    idx = load_faiss()
    names = load_names()
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not files:
        print("No images found in images/ folder. Add files like 'john_doe.jpg' and retry.")
        return
    for fn in files:
        path = os.path.join(images_dir, fn)
        name = name_from_filename(fn)
        print(f"Processing: {fn} -> Name: {name}")
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read {path}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = get_embedding(app, img)
        if emb is None:
            print(f"No face detected in {fn}, skipping.")
            continue
        emb = l2_normalize(emb)
        # check if name already exists: if so, append; else create new id
        existing_id = None
        for pid, pname in names.items():
            if pname.lower() == name.lower():
                existing_id = pid
                break
        if existing_id is None:
            pid = next_id(names)
            names[pid] = name
        else:
            pid = existing_id
        # add embedding with id
        vec = emb.reshape(1, -1).astype('float32')
        ids = np.array([pid], dtype='int64')
        idx.add_with_ids(vec, ids)
        print(f"Added embedding for {name} with id={pid}")
    save_faiss(idx)
    save_names(names)
    print("Registration complete. FAISS index and names saved to database/ folder.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default=None, help='Optional path to images folder (defaults to ./images)')
    args = parser.parse_args()
    images_dir = args.images if args.images else IMAGES_DIR
    print("Loading InsightFace models (may download on first run)...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640,640))
    register_all(app, images_dir)

if __name__ == '__main__':
    main()
