import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load('model/mnist_cnn.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_digit(input_data):
    if isinstance(input_data, dict):
        image_array = input_data.get("composite", input_data.get("image"))
    else:
        image_array = input_data

    if image_array is None:
        return {str(i): 0 for i in range(10)}

    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)

    bbox = image.getbbox()
    if bbox:
        cropped = image.crop(bbox)
        new_size = (28, 28)
        centered_image = Image.new("L", new_size, 0)
        paste_pos = (
            (new_size[0] - cropped.width) // 2,
            (new_size[1] - cropped.height) // 2
        )
        centered_image.paste(cropped, paste_pos)
    else:
        centered_image = Image.new("L", (28, 28), 0)

    image_tensor = transform(centered_image).unsqueeze(0)
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