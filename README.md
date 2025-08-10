# Face Recognition with InsightFace

A Python-based face recognition application using [InsightFace](https://github.com/deepinsight/insightface) for detecting and recognizing faces from images or webcam.  
Includes a Tkinter-based GUI for easy use.

---

## ✨ Features
- Register faces from images
- Recognize faces from:
  - Webcam (real-time)
  - Static images
- Adjustable similarity threshold
- Works offline after first model download
- Color-coded recognition results:
  - 🟢 Green = recognized (above threshold)
  - 🔴 Red = unknown (below threshold)

---

## 📂 Project Structure
face_recognition/
│
├── recognize.py # GUI + recognition logic
├── register_faces.py # Register new faces into the database
├── utils.py # Helper functions (FAISS, embeddings, normalization)
├── requirements.txt # Python dependencies
├── images/ # Folder for storing registered face images
├── README.md 
└── .gitignore # Ignore unnecessary files in Git



---

## 🛠 Installation (Windows)
1. **Clone this repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME


2. **Create and activate virtual environment**
python -m venv venv
venv\Scripts\activate


3. **Install dependencies**
pip install -r requirements.txt


📦 Dependencies
insightface
faiss-cpu
opencv-python
tkinter (built-in with Python on Windows)
numpy

4. **Registering Faces**
Place images of people you want to register in the images/ folder.
Each image should only contain one face.
File name (without extension) will be the person's name.
Run the registration script:
python register_faces.py

5. **🎯 Running Face Recognition**
python recognize.py --threshold 0.5
