# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150200111008_4 MUHAMMAD HERDI ADAM_4

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load('model/mnist_cnn.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image_array):
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)

    image = image.point(lambda x: 0 if x < 50 else 255, '1')

    bbox = image.getbbox()
    if bbox:
        cropped = image.crop(bbox)
        cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
        new_image = Image.new('L', (28, 28), 0)
        top_left = ((28 - cropped.width) // 2, (28 - cropped.height) // 2)
        new_image.paste(cropped, top_left)
    else:

        new_image = Image.new('L', (28, 28), 0)

    return new_image

def predict_digit(input_data):
    if isinstance(input_data, dict):
        image_array = input_data.get("composite", input_data.get("image"))
    else:
        image_array = input_data

    if image_array is None:
        return {str(i): 0.0 for i in range(10)}

    processed = preprocess_image(image_array)

    image_tensor = transform(processed).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predictions = {str(i): float(probabilities[0][i]) for i in range(10)}
        return predictions

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Gambar Digit Tulisan Tangan Anda", type="numpy", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=3, label="Hasil Prediksi"),
    title="Klasifikasi Digit Tulisan Tangan MNIST",
    description="Gambar sebuah digit dari 0-9 pada kanvas di bawah dan klik 'Submit' untuk melihat prediksi model."
)

iface.launch()
