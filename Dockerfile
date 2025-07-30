# Use official PyTorch image with CUDA 12.8
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# Copy project dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy project files
COPY . .

# Default command: start bash
CMD ["bash"]
