import tkinter as tk
import time
import math
import numpy as np
import threading
import cv2

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

class BiometricApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Mouse + Eye Biometric App")
        self.root.geometry("600x400")

        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.info_label.pack(pady=10)

        self.start_button = tk.Button(self.root, text="Start Trial 1", command=self.start_trial)
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

        self.info_label.config(text=f"Trial {self.trial} started.\nMove your mouse and keep face visible to camera.")
        self.root.after(10000, self.end_trial)

        threading.Thread(target=self.track_eyes, daemon=True).start()

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
            # Separate MSE
            mse_mouse = np.mean((self.trial1_data[:3] - feature_vector[:3]) ** 2)
            mse_eye = np.mean((self.trial1_data[3:] - feature_vector[3:]) ** 2)

            # Convert to similarity %
            mouse_similarity = max(0, 100 - mse_mouse / 5)  # scale factor adjustable
            eye_similarity = max(0, 100 - mse_eye / 5)

            # Weighted overall decision
            mouse_weight = 0.7
            eye_weight = 0.3
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

# Run the app
if __name__ == "_main_":
    root = tk.Tk()
    app = BiometricApp(root)
    root.mainloop()