# WhisperX API con FastAPI y Docker (GPU)

Esta API proporciona transcripción de audio utilizando WhisperX, con soporte para alineación a nivel de palabra y diarización de hablantes. Está **optimizada para ejecutarse en GPUs NVIDIA**.

## Requisitos Previos

*   **Docker:** [Instrucciones de instalación](https://docs.docker.com/get-docker/)
*   **Docker Compose** (v1.28+ o Docker Compose V2): [Instrucciones de instalación](https://docs.docker.com/compose/install/)
*   **Drivers NVIDIA** actualizados en el sistema host.
*   **NVIDIA Container Toolkit** instalado en el sistema host: [Instrucciones](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Esto permite a los contenedores Docker acceder a las GPUs NVIDIA.
*   (Opcional, para diarización) Un **token de acceso de Hugging Face**: [Obtener token](https://huggingface.co/docs/hub/security-tokens).
    *   Deberás aceptar los términos de uso de los modelos `pyannote/speaker-diarization-3.1` y `pyannote/segmentation-3.1` (o las versiones que uses) en Hugging Face.

## Estructura del Repositorio