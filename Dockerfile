# =================================================================
# --- TAHAP 1: "BUILDER" ---
# Tahap ini bertugas untuk melatih model dan menghasilkan file .pth
# =================================================================
FROM python:3.10-slim as builder

# Menetapkan direktori kerja untuk tahap build
WORKDIR /build

# Menyalin dan menginstal dependensi untuk training
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin skrip training
COPY train.py .

# Menjalankan skrip training untuk menghasilkan 'model/mnist_cnn.pth'
RUN python train.py


# =================================================================
# --- TAHAP 2: "FINAL APP" ---
# Tahap ini akan menjadi image final yang bersih dan siap jalan
# =================================================================
FROM python:3.10-slim

# Menetapkan direktori kerja untuk aplikasi
WORKDIR /app

# Menyalin dan menginstal dependensi HANYA untuk aplikasi Gradio
COPY App/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin kode aplikasi
COPY App/ ./App
# Kita masih perlu train.py untuk mengimpor kelas SimpleCNN
COPY train.py .

# --- Perintah Kunci ---
# Menyalin HANYA folder 'model' yang berisi file .pth dari tahap 'builder'
COPY --from=builder /build/model/ ./model

# Memberitahu Docker port mana yang akan diekspos
EXPOSE 7860

# Perintah untuk menjalankan aplikasi
CMD ["python", "App/app.py"]