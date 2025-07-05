import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from train import SimpleCNN

# load model yang sudah dilatih
model = SimpleCNN()
model.load_state_dict(torch.load('../model/mnist_cnn.pth'))
model.eval()

# transform untuk input gambar
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_digit(image_array):
    """Fungsi untuk memproses input gambar dan memberikan prediksi."""
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = probabilities.topk(1, dim=1)
        
        predictions = {str(i): float(probabilities[0][i]) for i in range(10)}
        return predictions

# Buat Gradio
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Gambar Digit Tulisan Tangan Anda", shape=(280, 280), image_mode="RGB", invert_colors=True),
    outputs=gr.Label(num_top_classes=3, label="Hasil Prediksi"),
    title="Klasifikasi Digit Tulisan Tangan MNIST",
    description="Gambar sebuah digit dari 0-9 pada kanvas di bawah dan klik 'Submit' untuk melihat prediksi model."
)

iface.launch()