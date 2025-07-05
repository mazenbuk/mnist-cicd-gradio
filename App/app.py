import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Tambahkan direktori root proyek ke path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Impor definisi kelas SimpleCNN dari train.py
from train import SimpleCNN

# Muat model yang sudah dilatih
model = SimpleCNN()
model.load_state_dict(torch.load('model/mnist_cnn.pth'))
model.eval()

# Transformasi untuk input gambar
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- PERBAIKAN UTAMA ADA DI FUNGSI INI ---
def predict_digit(input_data):
    """Fungsi untuk memproses input gambar dan memberikan prediksi."""
    
    # Logika baru untuk menangani berbagai format output dari Gradio
    if isinstance(input_data, dict):
        # Di versi Gradio yang lebih baru, outputnya bisa berupa dict dengan kunci 'composite'
        image_array = input_data.get("composite", input_data.get("image"))
    else:
        # Jika input bukan dictionary, asumsikan itu adalah array gambar langsung
        image_array = input_data

    if image_array is None:
        return {str(i): 0 for i in range(10)}
        
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predictions = {str(i): float(probabilities[0][i]) for i in range(10)}
        return predictions

# Buat Antarmuka Gradio
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Gambar Digit Tulisan Tangan Anda", type="numpy", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=3, label="Hasil Prediksi"),
    title="Klasifikasi Digit Tulisan Tangan MNIST",
    description="Gambar sebuah digit dari 0-9 pada kanvas di bawah dan klik 'Submit' untuk melihat prediksi model."
)

# Luncurkan aplikasi
iface.launch()