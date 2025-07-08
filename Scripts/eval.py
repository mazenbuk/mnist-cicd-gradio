# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'Model')
RESULTS_DIR = os.path.join(ROOT_DIR, 'Results')
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

os.makedirs(RESULTS_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'mnist_cnn.pt')
model = torch.jit.load(model_path, map_location='cpu')
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
print(f"Metrics saved to {metrics_path}")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on MNIST Test Set')
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, 'results.png')
plt.savefig(plot_path)
plt.close()
print(f"Confusion matrix plot saved to {plot_path}")