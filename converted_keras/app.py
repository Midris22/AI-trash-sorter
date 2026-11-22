import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from predict_webcam import SCRIPT_DIR, load_labels, load_model, predict_frame


BACKGROUND = "#F3F6F2"
SURFACE = "#FFFFFF"
INK = "#18251E"
MUTED = "#718078"
GREEN = "#26734D"
GREEN_DARK = "#1D5A3C"
GREEN_PALE = "#E8F2E9"
BORDER = "#DDE6DE"
PREVIEW_BACKGROUND = "#F7F9F7"
PREVIEW_WIDTH = 400
PREVIEW_HEIGHT = 230
IMAGE_FILE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def classify_image_file(model, image_path, labels):
    with Image.open(image_path) as source:
        rgb_frame = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))

    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    predicted_class, confidence, ai_input = predict_frame(model, bgr_frame, labels)
    return rgb_frame, predicted_class, confidence, ai_input


class TrashSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trash Sorter")
        self.root.geometry("1040x800")
        self.root.minsize(980, 780)
        self.root.configure(bg=BACKGROUND)

        self.model = None
        self.labels = None
        self.closed = False
        self.events = queue.Queue()
        self.status = tk.StringVar(value="Loading your model...")
        self.prediction = tk.StringVar(value="Ready for an image")
        self.confidence = tk.StringVar(value="Your result will appear here")

        content = tk.Frame(root, bg=BACKGROUND, padx=32, pady=26)
        content.pack(fill="both", expand=True)

        header = tk.Frame(content, bg=BACKGROUND)
        header.pack(fill="x", pady=(0, 22))
        heading = tk.Frame(header, bg=BACKGROUND)
        heading.pack(side="left", anchor="w")
        tk.Label(
            heading,
            text="AI TRASH SORTER",
            bg=BACKGROUND,
            fg=GREEN,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        tk.Label(
            heading,
            text="Sort smarter.",
            bg=BACKGROUND,
            fg=INK,
            font=("Segoe UI", 28, "bold"),
        ).pack(anchor="w")
        tk.Label(
            heading,
            text="Choose a photo and let AI identify the material.",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(4, 0))

        self.model_badge = tk.Label(
            header,
            text="LOADING MODEL",
            bg="#FFF1D6",
            fg="#865D13",
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=8,
        )
        self.model_badge.pack(side="right", anchor="n", pady=6)

        previews = tk.Frame(content, bg=BACKGROUND)
        previews.pack(fill="x")
        self.file_canvas, self.file_image_id, self.file_placeholder_id = self.create_preview(
            previews,
            eyebrow="01  /  SOURCE",
            title="Your image",
            description="The photo you selected",
            placeholder="Choose an image to get started",
        )
        self.ai_canvas, self.ai_image_id, self.ai_placeholder_id = self.create_preview(
            previews,
            eyebrow="02  /  MODEL VIEW",
            title="What the AI sees",
            description="Resized to 224 x 224, shown in normal colors",
            placeholder="The model's view will appear here",
        )

        actions = tk.Frame(content, bg=BACKGROUND)
        actions.pack(fill="x", pady=(18, 16))
        self.choose_button = tk.Button(
            actions,
            text="Choose an image",
            command=self.choose_image,
            state="disabled",
            bg=GREEN,
            fg="#FFFFFF",
            activebackground=GREEN_DARK,
            activeforeground="#FFFFFF",
            disabledforeground="#E8F0E9",
            relief="flat",
            borderwidth=0,
            padx=24,
            pady=12,
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        )
        self.choose_button.pack(side="left")
        self.file_name = tk.Label(
            actions,
            text="PNG, JPG, BMP, GIF, TIFF, or WEBP",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 9),
            padx=12,
        )
        self.file_name.pack(side="left")

        result = tk.Frame(
            content,
            bg=SURFACE,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=22,
            pady=18,
        )
        result.pack(fill="x")

        result_text = tk.Frame(result, bg=SURFACE)
        result_text.pack(side="left", fill="x", expand=True)
        tk.Label(
            result_text,
            text="PREDICTION",
            bg=SURFACE,
            fg=GREEN,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        tk.Label(
            result_text,
            textvariable=self.prediction,
            bg=SURFACE,
            fg=INK,
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w")

        confidence_panel = tk.Frame(result, bg=SURFACE, width=290)
        confidence_panel.pack(side="right", padx=(24, 0), anchor="center")
        confidence_panel.pack_propagate(False)
        tk.Label(
            confidence_panel,
            textvariable=self.confidence,
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 10))
        self.confidence_bar = tk.Canvas(
            confidence_panel,
            width=280,
            height=10,
            bg=SURFACE,
            highlightthickness=0,
        )
        self.confidence_bar.pack(anchor="w")
        self.confidence_bar.create_rectangle(
            0, 0, 280, 10, fill=GREEN_PALE, outline="", tags="track"
        )
        self.confidence_fill = self.confidence_bar.create_rectangle(
            0, 0, 0, 10, fill=GREEN, outline="", tags="fill"
        )

        tk.Label(
            content,
            textvariable=self.status,
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(12, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.poll_events)
        threading.Thread(target=self.load_model_in_background).start()

    def create_preview(self, parent, eyebrow, title, description, placeholder):
        card = tk.Frame(
            parent,
            bg=SURFACE,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=18,
            pady=16,
        )
        card.pack(side="left", fill="both", expand=True, padx=(0, 12) if eyebrow.startswith("01") else (0, 0))

        tk.Label(
            card,
            text=eyebrow,
            bg=SURFACE,
            fg=GREEN,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            text=title,
            bg=SURFACE,
            fg=INK,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", pady=(5, 1))
        tk.Label(
            card,
            text=description,
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 12))

        canvas = tk.Canvas(
            card,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            bg=PREVIEW_BACKGROUND,
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)
        image_id = canvas.create_image(PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2)
        placeholder_id = canvas.create_text(
            PREVIEW_WIDTH // 2,
            PREVIEW_HEIGHT // 2,
            text=placeholder,
            fill=MUTED,
            font=("Segoe UI", 10),
            width=300,
        )
        return canvas, image_id, placeholder_id

    def load_model_in_background(self):
        try:
            model = load_model()
            labels = load_labels(SCRIPT_DIR / "labels.txt")
            self.events.put(("model_ready", (model, labels)))
        except Exception as error:
            self.events.put(("model_error", str(error)))

    def choose_image(self):
        selected_path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose an image to classify",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not selected_path:
            return

        image_path = Path(selected_path)
        self.choose_button.configure(state="disabled", cursor="arrow")
        self.prediction.set("Analyzing image...")
        self.confidence.set("Calculating confidence...")
        self.confidence_bar.coords(self.confidence_fill, 0, 0, 0, 10)
        self.file_name.configure(text=image_path.name, fg=INK)
        self.status.set("Reading image and running the model...")
        threading.Thread(
            target=self.classify_image_in_background,
            args=(image_path,),
        ).start()

    def classify_image_in_background(self, image_path):
        try:
            rgb_frame, predicted_class, confidence, ai_input = classify_image_file(
                self.model,
                image_path,
                self.labels,
            )
            self.events.put(
                ("prediction", (image_path.name, rgb_frame, predicted_class, confidence, ai_input))
            )
        except Exception as error:
            self.events.put(("prediction_error", (image_path.name, str(error))))

    def make_photo(self, frame):
        height, width = frame.shape[:2]
        scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height)
        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))
        resized = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(resized), master=self.root)

    def update_preview(self, canvas, image_id, placeholder_id, frame):
        photo = self.make_photo(frame)
        canvas.itemconfigure(image_id, image=photo)
        canvas.itemconfigure(placeholder_id, state="hidden")
        return photo

    def update_confidence_bar(self, confidence):
        confidence = min(max(confidence, 0.0), 1.0)
        self.confidence_bar.coords(self.confidence_fill, 0, 0, 280 * confidence, 10)

    def poll_events(self):
        if self.closed:
            return

        try:
            while True:
                event_type, payload = self.events.get_nowait()
                if event_type == "model_ready":
                    self.model, self.labels = payload
                    self.choose_button.configure(state="normal", cursor="hand2")
                    self.model_badge.configure(
                        text="MODEL READY",
                        bg=GREEN_PALE,
                        fg=GREEN_DARK,
                    )
                    self.status.set("Model ready. Choose an image to begin.")
                elif event_type == "model_error":
                    self.model_badge.configure(
                        text="MODEL ERROR",
                        bg="#FCE9E7",
                        fg="#A13B32",
                    )
                    self.status.set(f"Could not load model: {payload}")
                elif event_type == "prediction":
                    image_name, rgb_frame, predicted_class, confidence, ai_input = payload
                    self.file_photo = self.update_preview(
                        self.file_canvas,
                        self.file_image_id,
                        self.file_placeholder_id,
                        rgb_frame,
                    )
                    self.ai_photo = self.update_preview(
                        self.ai_canvas,
                        self.ai_image_id,
                        self.ai_placeholder_id,
                        ai_input,
                    )
                    self.prediction.set(predicted_class)
                    self.confidence.set(f"{confidence:.1%} confidence")
                    self.update_confidence_bar(confidence)
                    self.status.set(f"Classified {image_name}")
                    self.choose_button.configure(state="normal", cursor="hand2")
                elif event_type == "prediction_error":
                    image_name, error = payload
                    self.status.set(f"Could not classify {image_name}: {error}")
                    self.choose_button.configure(state="normal", cursor="hand2")
        except queue.Empty:
            pass

        self.root.after(100, self.poll_events)

    def close(self):
        self.closed = True
        self.root.destroy()


def main():
    root = tk.Tk()
    TrashSorterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
