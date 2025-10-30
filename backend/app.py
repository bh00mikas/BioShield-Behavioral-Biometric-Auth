import tkinter as tk
import time
import math
import numpy as np
from tkinter import messagebox

class MouseBiometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Biometric Verification")
        self.root.geometry("600x400")

        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(self.root, text="", font=("Helvetica", 12), justify="left")
        self.info_label.pack(pady=10)

        self.start_button = tk.Button(self.root, text="Start Trial 1", command=self.start_trial)
        self.start_button.pack(pady=10)

        self.reset_tracking()

        self.trial = 1
        self.trial1_features = None

        self.canvas.bind("<Motion>", self.track_mouse)

    def reset_tracking(self):
        self.last_time = None
        self.last_pos = None
        self.velocities = []
        self.accelerations = []
        self.angles = []
        self.start_time = None

    def start_trial(self):
        self.reset_tracking()
        self.start_time = time.time()
        self.info_label.config(text=f"Trial {self.trial} started. Move your mouse on the canvas for 10 seconds.")
        self.root.after(10000, self.end_trial)

    def end_trial(self):
        velocity_avg = np.mean(self.velocities) if self.velocities else 0
        accel_avg = np.mean(self.accelerations) if self.accelerations else 0
        angle_std = np.std(self.angles) if self.angles else 0

        features = np.array([velocity_avg, accel_avg, angle_std])

        if self.trial == 1:
            self.trial1_features = features
            self.info_label.config(text=f"Trial 1 completed.\nFeatures recorded.\nNow start Trial 2.")
            self.start_button.config(text="Start Trial 2")
            self.trial += 1
        else:
            diff = np.square(self.trial1_features - features)
            mse = np.mean(diff)

            threshold = 1000  # You can tune this threshold
            result = "Same user ✅" if mse < threshold else "Different user ❌"

            self.info_label.config(
                text=f"Trial 2 completed.\n\nSimilarity score (MSE): {mse:.2f}\n\nResult: {result}"
            )
            self.start_button.config(state="disabled")

    def track_mouse(self, event):
        if not self.start_time:
            return

        current_time = time.time()
        current_pos = (event.x, event.y)

        if self.last_pos and self.last_time:
            dt = current_time - self.last_time
            dx = current_pos[0] - self.last_pos[0]
            dy = current_pos[1] - self.last_pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            if dt > 0:
                velocity = distance / dt
                self.velocities.append(velocity)

                if len(self.velocities) > 1:
                    acceleration = (velocity - self.velocities[-2]) / dt
                    self.accelerations.append(acceleration)

                angle = math.degrees(math.atan2(dy, dx))
                self.angles.append(angle)

        self.last_pos = current_pos
        self.last_time = current_time


if __name__ == "__main__":
    root = tk.Tk()
    app = MouseBiometricApp(root)
    root.mainloop()
