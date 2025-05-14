FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory inside the container
WORKDIR /app

# ✅ Install system packages including build tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*


# COPY greenstand_segmentation_model_build/segmentation_model_build/AutoSAM /app/AutoSAM
COPY sam_dino_pipeline_sagemaker/sam2/sam2_dino_docker/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY sam_dino_pipeline_sagemaker/sam2 /app/sam_dino_pipeline_sagemaker/sam2

WORKDIR /app/sam_dino_pipeline_sagemaker/sam2
RUN pip install -e .

# ENTRYPOINT
COPY sam_dino_pipeline_sagemaker/sam2/sam2_dino_docker/sagemaker_entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/sagemaker_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
