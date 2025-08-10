import os, cv2, numpy as np, argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from insightface.app import FaceAnalysis
from utils import load_faiss, load_names, l2_normalize
import time

DEFAULT_THRESHOLD = 0.50  # cosine-like similarity threshold (adjustable)

def recognize_image_file(app, idx, names, filepath, threshold=DEFAULT_THRESHOLD):
    img = cv2.imread(filepath)
    if img is None:
        messagebox.showerror("Error", f"Cannot open image: {filepath}")
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    if not faces:
        messagebox.showinfo("Result", "No faces detected in the image.")
        return
    out = img.copy()
    results = []
    for face in faces:
        x1,y1,x2,y2 = map(int, face.bbox[:4])
        emb = l2_normalize(np.array(face.normed_embedding, dtype='float32'))
        D, I = idx.search(emb.reshape(1, -1), 5)
        best_name = "Unknown"
        best_sim = -1.0
        for dist, pid in zip(D[0], I[0]):
            if pid == -1:
                continue
            sim = 1 - dist/2.0
            if sim > best_sim:
                best_sim = sim
                best_name = names.get(int(pid), "Unknown")
        # label = f"{best_name} ({best_sim:.2f})" if best_sim>=0 else best_name
        # color = (0,255,0) if best_sim >= threshold else (0,0,255)
        if best_sim >= threshold:
            label = f"{best_name} ({best_sim:.2f})"
            color = (0,255,0)  # green for match
        else:
            label = f"Unknown ({best_sim:.2f})"
            color = (0,0,255)  # red for no match
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        results.append((best_name, best_sim))
    # show image in a window
    cv2.imshow('Recognition - Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_webcam(app, idx, names, threshold=DEFAULT_THRESHOLD):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # use DirectShow on Windows for better compatibility
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not accessible. Check camera and permissions.")
        return
    win_name = 'Recognition - Webcam (press q to quit)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)
        display = frame.copy()
        if faces:
            for face in faces:
                x1,y1,x2,y2 = map(int, face.bbox[:4])
                emb = l2_normalize(np.array(face.normed_embedding, dtype='float32'))
                D, I = idx.search(emb.reshape(1, -1), 5)
                best_name = "Unknown"
                best_sim = -1.0
                for dist, pid in zip(D[0], I[0]):
                    if pid == -1:
                        continue
                    sim = 1 - dist/2.0
                    if sim > best_sim:
                        best_sim = sim
                        best_name = names.get(int(pid), "Unknown")
                # label = f"{best_name} ({best_sim:.2f})" if best_sim>=0 else best_name
                # color = (0,255,0) if best_sim >= threshold else (0,0,255)
                if best_sim >= threshold:
                    label = f"{best_name} ({best_sim:.2f})"
                    color = (0,255,0)  # green for match
                else:
                    label = f"Unknown ({best_sim:.2f})"
                    color = (0,0,255)  # red for no match
                cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow(win_name, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def on_webcam_click(app, idx, names, threshold_var):
    try:
        recognize_webcam(app, idx, names, threshold=float(threshold_var.get()))
    except Exception as e:
        messagebox.showerror("Error", str(e))

def on_image_click(app, idx, names, threshold_var):
    path = filedialog.askopenfilename(title='Select image for recognition', filetypes=[('Image files','*.jpg *.jpeg *.png')])
    if not path:
        return
    try:
        recognize_image_file(app, idx, names, path, threshold=float(threshold_var.get()))
    except Exception as e:
        messagebox.showerror("Error", str(e))

def build_gui(app, idx, names):
    root = tk.Tk()
    root.title('Face Recognition - Windows')
    root.geometry('360x160')
    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))

    tk.Label(frame, text='Similarity threshold (0..1):').grid(row=0, column=0, sticky='w')
    tk.Entry(frame, textvariable=threshold_var, width=10).grid(row=0, column=1, sticky='e')

    btn1 = tk.Button(frame, text='Recognize from Webcam', width=25, command=lambda: on_webcam_click(app, idx, names, threshold_var))
    btn1.grid(row=1, column=0, columnspan=2, pady=8)

    btn2 = tk.Button(frame, text='Recognize from Image', width=25, command=lambda: on_image_click(app, idx, names, threshold_var))
    btn2.grid(row=2, column=0, columnspan=2, pady=8)

    tk.Label(frame, text='Tip: Press q in webcam window to quit.').grid(row=3, column=0, columnspan=2, pady=4)

    root.mainloop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Default similarity threshold')
    args = parser.parse_args()

    print("Loading models (may download first time)...") 
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640,640))

    idx = load_faiss()
    names = load_names()
    if not names:
        print("Warning: No registered faces found. Run register_faces.py first to add images from images/ folder.")
    build_gui(app, idx, names)

if __name__ == '__main__':
    main()
