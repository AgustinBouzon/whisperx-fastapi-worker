# WhisperX API con FastAPI y Docker (GPU)

Esta API proporciona transcripción de audio utilizando WhisperX, con soporte para alineación a nivel de palabra y diarización de hablantes. Está **optimizada para ejecutarse en GPUs NVIDIA**.

## Requisitos Previos

- **Docker:** [Instrucciones de instalación](https://docs.docker.com/get-docker/)
- **Docker Compose** (v1.28+ o Docker Compose V2): [Instrucciones de instalación](https://docs.docker.com/compose/install/)
- **Drivers NVIDIA** actualizados en el sistema host.
- **NVIDIA Container Toolkit** instalado en el sistema host: [Instrucciones](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Esto permite a los contenedores Docker acceder a las GPUs NVIDIA.
- (Opcional, para diarización) Un **token de acceso de Hugging Face**: [Obtener token](https://huggingface.co/docs/hub/security-tokens).
  - Deberás aceptar los términos de uso de los modelos `pyannote/speaker-diarization-3.1` y `pyannote/segmentation-3.1` (o las versiones que uses) en Hugging Face.

## Estructura del Repositorio

```
whisperx-api-gpu/
├── .gitignore
├── app/
│   ├── __init__.py
│   └── main.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

## Configuración y Ejecución

1.  **Clona este repositorio.**

2.  **Prepara los directorios de caché (recomendado para persistencia de modelos):**
    En la raíz del proyecto, crea los directorios donde se guardarán los modelos descargados:

    ```bash
    mkdir -p ./model_cache/whisper
    mkdir -p ./model_cache/alignment
    mkdir -p ./model_cache/huggingface
    # mkdir -p ./model_cache/torch_pyannote # Opcional, dependiendo de dónde guarde exactamente pyannote
    ```

    La aplicación está configurada para utilizar estos directorios específicos para almacenar en caché los modelos de transcripción de Whisper (`./model_cache/whisper`), los modelos de alineación (`./model_cache/alignment`), y los modelos descargados de Hugging Face como Pyannote (`./model_cache/huggingface`).

3.  **Token de Hugging Face para Diarización (Opcional pero necesario para diarización):**
    Si planeas usar la diarización de hablantes:

    - Abre el archivo `docker-compose.yml`.
    - Busca la sección `environment` y descomenta la línea `HF_TOKEN`.
    - Reemplaza `"tu_hugging_face_token_aqui"` con tu token real.

    ```yaml
    environment:
      HF_TOKEN: "hf_xxxxYOURTOKENxxxx" # Reemplaza con tu token
      # ...
    ```

    Alternativamente, puedes pasar el token en cada request a la API.

4.  **Construir la imagen Docker:**
    Navega al directorio raíz del proyecto y ejecuta:

    ```bash
    docker-compose build
    ```

    Esto utilizará el `Dockerfile` para construir una imagen llamada `whisperx_api_gpu_service` (o similar, basado en el nombre del directorio del proyecto si `container_name` no está definido en compose).

5.  **Ejecutar el contenedor:**

    ```bash
    docker-compose up -d
    ```

    Esto iniciará el servicio en segundo plano. Docker Compose se encargará de pasar las capacidades de GPU al contenedor.

6.  **Verificar que la API está funcionando y usando GPU:**

    - **Logs del contenedor:**

      ```bash
      docker-compose logs -f whisperx_api_gpu
      ```

      Deberías ver mensajes indicando que se está usando el dispositivo `"cuda"`, la versión de PyTorch, CUDA, etc.

    - **Endpoint de salud:**
      Abre tu navegador o usa `curl`:

      ```bash
      curl http://localhost:8000/health
      ```

      Debería devolver algo como:
      `{"status":"ok","device_available":"cuda","torch_version":"2.1.2+cu121","cuda_available":true}`

    - **Documentación Interactiva (Swagger UI):**
      ```
      http://localhost:8000/docs
      ```

## Uso de la API

### Endpoint: `POST /transcribe/`

Realiza una petición POST a `http://localhost:8000/transcribe/` con `multipart/form-data`.

**Parámetros (form-data):**

- `audio_file` (requerido): El archivo de audio (ej. `.wav`, `.mp3`).
- `model_name` (opcional, default: `"base"`): Modelo de Whisper a usar (ej: `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v1"`, `"large-v2"`, `"large-v3"`). Modelos más grandes requieren más VRAM.
- `language` (opcional, default: `None` - autodetección): Código de idioma de dos letras (ej: `"es"`, `"en"`). Para `large-v3`, especificar el idioma puede mejorar la calidad.
- `batch_size` (opcional, default: `16`): Tamaño del lote para transcripción. WhisperX recomienda 4, 8, o 16 para GPU. Ajusta según tu VRAM y el modelo.
- `align_audio` (opcional, default: `True`): `true` o `false` para habilitar/deshabilitar alineación de palabras.
- `diarize_audio` (opcional, default: `False`): `true` o `false` para habilitar/deshabilitar diarización de hablantes.
- `hf_token` (opcional, default: `None`): Token de Hugging Face si `diarize_audio` es `true` y no se proveyó mediante `docker-compose.yml`.
- `min_speakers` (opcional, default: `None`): Número mínimo de hablantes para diarización (pasado a `pyannote.audio`).
- `max_speakers` (opcional, default: `None`): Número máximo de hablantes para diarización (pasado a `pyannote.audio`).

**Ejemplo con `curl`:**

```bash
curl -X POST "http://localhost:8000/transcribe/" \
  -F "audio_file=@/ruta/a/tu/audio.wav" \
  -F "model_name=medium" \
  -F "language=en" \
  -F "align_audio=true" \
  -F "diarize_audio=true" # Si no pusiste HF_TOKEN en docker-compose, añádelo aquí: -F "hf_token=tu_token"
```
