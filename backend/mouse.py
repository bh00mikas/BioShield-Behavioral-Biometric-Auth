# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:09:33 2025

@author: tanma
"""

import tkinter as tk
import time
import math
import numpy as np

class MouseBiometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Biometric App")
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
