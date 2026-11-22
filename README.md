# AI Trash Sorter

A desktop app that classifies an image as Paper, Plastic, Compost, or Trash.
The app shows both the selected image and the resized image input used by the
model.

## Setup

From the project directory, create the virtual environment and install the
dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe .\converted_keras\app.py
```

Choose an image in the app to see its predicted class and confidence. The model
and class labels are loaded from `converted_keras`.
