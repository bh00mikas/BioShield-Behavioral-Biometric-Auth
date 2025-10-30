import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import time
import math
import numpy as np
import threading
import cv2

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# --- Helper Functions ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return (dot / (norm1 * norm2)) if norm1 and norm2 else 0

# --- Biometric GUI App ---
class BiometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse + Eye Biometric Authentication")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f4f7")

        self.canvas = tk.Canvas(self.root, bg="#ffffff", height=300)
        self.canvas.pack(pady=20, padx=20, fill=tk.BOTH, expand=False)

        self.images_frame = tk.Frame(self.root, bg="#f0f4f7")
        self.images_frame.pack(pady=10)

        # Load and store image references
        self.eye_img = Image.open("eye.png").resize((100, 100))
        self.mouse_img = Image.open("mouse.png").resize((100, 100))
        self.eye_photo = ImageTk.PhotoImage(self.eye_img)
        self.mouse_photo = ImageTk.PhotoImage(self.mouse_img)
        self.image_refs = [self.eye_photo, self.mouse_photo]

        self.eye_label = tk.Label(self.images_frame, image=self.eye_photo, bg="#f0f4f7",
                                  text="Eye Tracking", compound="top", font=("Arial", 10, "bold"))
        self.mouse_label = tk.Label(self.images_frame, image=self.mouse_photo, bg="#f0f4f7",
                                    text="Mouse Tracking", compound="top", font=("Arial", 10, "bold"))
        self.eye_label.pack(side=tk.LEFT, padx=20)
        self.mouse_label.pack(side=tk.RIGHT, padx=20)

        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#f0f4f7", fg="#333")
        self.info_label.pack(pady=10)

        self.start_button = tk.Button(self.root, text="Start Trial 1", command=self.start_trial,
                                      bg="#4caf50", fg="white", font=("Arial", 12), width=20)
        self.start_button.pack(pady=10)

        self.canvas.bind("<Motion>", self.track_mouse)

        self.reset_mouse()
        self.trial = 1
        self.trial1_data = None
        self.eye_data = []
        self.running = False

    def reset_mouse(self):
        self.velocities = []
        self.accelerations = []
        self.angles = []
        self.last_time = None
        self.last_pos = None

    def track_mouse(self, event):
        if not self.running:
            return
        current_time = time.time()
        current_pos = (event.x, event.y)
        if self.last_pos and self.last_time:
            dt = current_time - self.last_time
            dx = current_pos[0] - self.last_pos[0]
            dy = current_pos[1] - self.last_pos[1]
            distance = math.hypot(dx, dy)
            if dt > 0:
                v = distance / dt
                self.velocities.append(v)
                if len(self.velocities) > 1:
                    a = (v - self.velocities[-2]) / dt
                    self.accelerations.append(a)
                angle = math.degrees(math.atan2(dy, dx))
                self.angles.append(angle)
        self.last_pos = current_pos
        self.last_time = current_time

    def start_trial(self):
        self.reset_mouse()
        self.eye_data = []
        self.running = True
        self.start_time = time.time()

        self.info_label.config(text=f"Trial {self.trial} started.\nMove mouse and keep face visible.")
        self.root.after(20000, self.end_trial)

        threading.Thread(target=self.track_eyes, daemon=True).start()

    def track_eyes(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        while time.time() - start_time < 20 and self.running:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:
                    eye_centers = [(ex + ew // 2, ey + eh // 2) for (ex, ey, ew, eh) in eyes[:2]]
                    avg_x = np.mean([ec[0] for ec in eye_centers])
                    avg_y = np.mean([ec[1] for ec in eye_centers])
                    self.eye_data.append([avg_x, avg_y])
                    break
        cap.release()

    def end_trial(self):
        self.running = False

        mouse_vector = np.array([
            np.mean(self.velocities) if self.velocities else 0,
            np.mean(self.accelerations) if self.accelerations else 0,
            np.std(self.angles) if self.angles else 0,
        ])
        eye_vector = np.mean(self.eye_data, axis=0) if self.eye_data else np.zeros(2)
        feature_vector = np.concatenate((mouse_vector, eye_vector))

        if self.trial == 1:
            self.trial1_data = feature_vector
            self.trial += 1
            self.start_button.config(text="Start Trial 2")
            self.info_label.config(text="Trial 1 complete.\nClick to start Trial 2.")
        else:
            mouse_vec_1 = normalize(self.trial1_data[:3])
            mouse_vec_2 = normalize(feature_vector[:3])
            mouse_similarity = cosine_similarity(mouse_vec_1, mouse_vec_2) * 100

            mse_eye = np.mean((self.trial1_data[3:] - feature_vector[3:]) ** 2)
            eye_similarity = max(0, 100 - mse_eye / 5)

            mouse_weight = 0.3
            eye_weight = 0.7
            total_similarity = mouse_weight * mouse_similarity + eye_weight * eye_similarity

            result = "Same user ✅" if total_similarity >= 70 else "Different user ❌"
            self.info_label.config(
                text=f"Trial 2 complete.\n"
                     f"Mouse Similarity: {mouse_similarity:.2f}%\n"
                     f"Eye Gaze Similarity: {eye_similarity:.2f}%\n"
                     f"Overall Match: {total_similarity:.2f}%\n"
                     f"Result: {result}"
            )
            self.start_button.config(state="disabled")

# --- Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = BiometricApp(root)
    root.mainloop()