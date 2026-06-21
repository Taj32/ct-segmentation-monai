
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# set directory in container
WORKDIR /app

# System-Level Dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py .
COPY download_model.py .

# Download the model
RUN python download_model.py

# Streamlit will listen at this HG Space Port
EXPOSE 7860

# Start Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"] 
