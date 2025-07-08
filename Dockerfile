# =================================================================
# --- FINAL APP ---
# This image will only run the Gradio app using the pre-trained model
# =================================================================
FROM python:3.10-slim

# Set the working directory for the application
WORKDIR /app

# Copy and install dependencies for the Gradio app
COPY App/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY App/ ./App

# Copy the pre-trained model from the 'Model' directory
COPY Model/ ./model

# Expose the port used by Gradio
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "App/app.py"]