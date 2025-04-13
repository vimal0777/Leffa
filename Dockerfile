# Base image with CUDA support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip python3.10-dev

# Create a symbolic link for "python" to point to "python3"
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Copy the project files into the container
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Download checkpoints (optional: can be skipped if checkpoints are pre-downloaded)
#RUN python app.py

# Expose the port for the Gradio interface
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]