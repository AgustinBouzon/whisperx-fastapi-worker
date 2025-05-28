# Etapa 1: Imagen base con PyTorch y CUDA
# Usar una imagen oficial de PyTorch que incluya CUDA y cuDNN.
# Elige una versión que sea compatible con los drivers de tu GPU host.
# Ejemplo: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# Puedes encontrar más tags en: https://hub.docker.com/r/pytorch/pytorch/tags
# Actualiza '2.1.0-cuda12.1-cudnn8-devel' según necesites.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel AS builder
# Alternativa: FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
# Alternativa: FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

LABEL maintainer="agustin.bouzon@linksolution.com.ar"
LABEL description="WhisperX API with FastAPI (GPU enabled)"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    # HuggingFace Hub cache (donde pyannote descarga modelos)
    HF_HOME="/app/model_cache/huggingface" \
    # Caché general (usado por whisperx para modelos de alineación, etc.)
    XDG_CACHE_HOME="/app/model_cache" \
    # Caché de modelos Whisper (whisperx los descarga aquí por defecto si no se especifica download_root)
    # Si whisperx usa ~/.cache/whisper, este se mapearía a XDG_CACHE_HOME/whisper
    # Para ser explícitos, podríamos configurar download_root en load_model, pero XDG_CACHE_HOME debería cubrirlo.
    PATH="/opt/conda/bin:${PATH}"
    # Asegurar que conda/pip de la imagen base estén en PATH

# Instalar dependencias del sistema: ffmpeg es crucial, git para instalar whisperx desde repo
# La imagen pytorch/pytorch ya es ubuntu-based y tiene muchas cosas.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Crear directorios para el caché de modelos (serán montados como volúmenes)
# Estos directorios deben coincidir con lo que esperan las bibliotecas o lo que se configura
RUN mkdir -p /app/model_cache/whisper \
             /app/model_cache/alignment \
             /app/model_cache/huggingface \
             /app/model_cache/torch_pyannote 
             # pyannote.audio guarda modelos de HF aquí por defecto (HF_HOME)

# Copiar requirements.txt primero para aprovechar el cache de Docker
COPY ./app/requirements.txt /app/requirements.txt

# Instalar dependencias de Python
# La imagen base de PyTorch ya tiene torch y torchaudio.
# Si requirements.txt lista versiones específicas, pip podría intentar reinstalarlas.
# Es mejor asegurarse de que las versiones en requirements.txt sean compatibles o genéricas (>=).
# El --no-deps para whisperx puede ser riesgoso si sus dependencias cambian mucho.
# Es mejor dejar que pip resuelva las dependencias.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN pip install git+https://github.com/m-bain/whisperX.git # Alternativa si no está en requirements

# Verificar la instalación de PyTorch y CUDA (opcional, para debugging)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')"

# Copiar el resto de la aplicación
COPY ./app /app

# Exponer el puerto en el que FastAPI se ejecutará
EXPOSE 8000

# Comando para ejecutar la aplicación
# Para GPU, 1 worker suele ser lo mejor para no competir por recursos de la GPU.
# Si tu GPU es muy potente o tienes múltiples GPUs y distribuyes el trabajo, podrías ajustar.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
