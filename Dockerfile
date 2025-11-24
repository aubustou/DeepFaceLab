# ser image NVIDIA avec CUDA 11.8 et cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Variables d’environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository universe 

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    build-essential \
    git \
    ffmpeg \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Lien python3 → python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Mettre à jour pip
RUN pip install --upgrade pip

# Cloner DeepFaceLab
WORKDIR /opt
RUN git clone --depth 1 https://github.com/aubustou/DeepFaceLab.git
WORKDIR /opt/DeepFaceLab

# Installer les dépendances Python
COPY requirements-cuda.txt .
RUN sed -i "s/numpy==1\.17\.3/numpy==1.23.5/" requirements-cuda.txt \
 && sed -i "s/scipy==1\.4\.1/scipy==1.10.1/" requirements-cuda.txt \
  && sed -i "s/h5py==2\.9\.0/h5py==3.8.0/" requirements-cuda.txt  \
   && sed -i "s/tensorflow-gpu==2\.4\.0/tensorflow==2.10.1/" requirements-cuda.txt \
    && sed -i "s/scikit-image==0\.14\.2/scikit-image==0.19.3/" requirements-cuda.txt

# Supprimer tf2onnx obsolète et incompatible
RUN sed -i "/tf2onnx/d" requirements-cuda.txt

RUN pip install -r requirements-cuda.txt

# Installer une version de tensorflow compatible
RUN pip install tensorflow==2.10.1   # ou version recommandée selon forks

# Définir workspace
ENV DFL_WORKSPACE="/workspace"
RUN mkdir -p ${DFL_WORKSPACE}

# Définir le dossier de modèle et data
VOLUME ["${DFL_WORKSPACE}"]

WORKDIR /workspace

# Commande par défaut
ENTRYPOINT ["python", "/opt/DeepFaceLab/main.py"]

