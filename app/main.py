import os
import uuid
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import torch
import whisperx
import logging
import collections

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhisperX API (GPU)",
    description="API para transcripción de audio usando WhisperX con alineación y diarización, optimizada para GPU.",
    version="0.1.0"
)

# Variables globales para los modelos
# loaded_models ahora es un OrderedDict para la gestión LRU de modelos Whisper
loaded_alignment_models = {}
loaded_diarization_pipelines = {}

# Determinar dispositivo (CUDA si está disponible, si no CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Usando dispositivo: {DEVICE}")
if DEVICE == "cuda":
    torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_avail else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    logger.info(f"PyTorch version: {torch_version}, CUDA disponible: {cuda_avail}, CUDA version: {cuda_version}, GPU: {gpu_name}")
else:
    logger.warning("CUDA no está disponible. La API se ejecutará en CPU, lo cual será significativamente más lento.")

# Configuración de MAX_WHISPER_MODELS
MAX_WHISPER_MODELS = int(os.environ.get("MAX_WHISPER_MODELS", "1")) # Default to 1 model
if MAX_WHISPER_MODELS <= 0: # Treat 0 or negative as unlimited (effectively disabling unloading)
    MAX_WHISPER_MODELS = float('inf')
logger.info(f"Maximum concurrent Whisper models allowed in memory: {MAX_WHISPER_MODELS if MAX_WHISPER_MODELS != float('inf') else 'Unlimited'}")

loaded_models = collections.OrderedDict() # Usar OrderedDict para LRU

# Tipos de cómputo recomendados para GPU y CPU
COMPUTE_TYPE_GPU = "float16"  # o "int8" para menor VRAM y mayor velocidad, con posible pérdida de precisión
COMPUTE_TYPE_CPU = "int8"     # o "float32" para CPU si "int8" da problemas

# --- Funciones auxiliares ---
def get_model(model_name: str, device: str, compute_type: str, language: str = None):
    model_key_suffix = f"_lang-{language}" if model_name == "large-v3" and language else ""
    model_key = (model_name + model_key_suffix, device, compute_type)

    if model_key in loaded_models:
        logger.info(f"Model {model_key} found in cache. Moving to end (most recently used).")
        loaded_models.move_to_end(model_key)
        return loaded_models[model_key]

    logger.info(f"Model {model_key} not found in cache. Attempting to load.")

    # Check if cache is full and needs eviction
    if len(loaded_models) >= MAX_WHISPER_MODELS:
        # Evict the least recently used model (first item in OrderedDict)
        oldest_key, oldest_model = loaded_models.popitem(last=False)
        logger.info(f"Cache is full (max_models={MAX_WHISPER_MODELS}). Evicting model {oldest_key} to free up memory.")
        try:
            # Attempt to clean up model resources
            # How to properly delete a whisperx model and free GPU memory can be intricate.
            # For PyTorch models, 'del model' and 'torch.cuda.empty_cache()' are standard.
            # WhisperX models might have internal CTranslate2 models or other components.
            # For now, we assume 'del' is sufficient for Python's garbage collector to work,
            # and empty_cache() helps with PyTorch's CUDA cache.
            # If whisperx.Model has a specific cleanup method, it should be called here.
            # (Assuming no specific cleanup method for now beyond Python's GC)
            del oldest_model
            if device == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Model {oldest_key} evicted and CUDA cache cleared (if on CUDA).")
        except Exception as e:
            logger.error(f"Error during eviction of model {oldest_key}: {e}", exc_info=True)

    # Load the new model
    logger.info(f"Loading Whisper model: {model_name} ({model_key_suffix.replace('_lang-','') if model_key_suffix else 'no lang specified'}) on {device} with compute_type {compute_type}")
    try:
        model_kwargs = {}
        if model_name == "large-v3" and language:
            model_kwargs['language'] = language

        # Explicitly set download_root to ensure it uses the mounted volume,
        # consistent with previous findings about XDG_CACHE_HOME and Whisper defaults.
        # The path should be /app/model_cache/whisper inside the container.
        download_root_path = "/app/model_cache/whisper" 
        # Ensure the directory exists (Docker should have created it, but good practice)
        # os.makedirs(download_root_path, exist_ok=True) # This might be too much for here, Dockerfile should handle.

        new_model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            download_root=download_root_path, # Explicitly use the cache path
            **model_kwargs
        )
        loaded_models[model_key] = new_model
        logger.info(f"Model {model_key} loaded and added to cache. Current cache size: {len(loaded_models)}.")
        return new_model
    except Exception as e:
        logger.error(f"Error loading Whisper model {model_key}: {e}", exc_info=True)
        # If model loading fails, ensure no partial entry is left in loaded_models for this key
        if model_key in loaded_models: # Should not happen if it's only added on success
             del loaded_models[model_key]
        raise HTTPException(status_code=500, detail=f"Error loading Whisper model {model_key}: {str(e)}")

def get_alignment_model(language_code: str, device: str):
    align_key = (language_code, device)
    if align_key not in loaded_alignment_models:
        logger.info(f"Cargando modelo de alineación para idioma: {language_code} en {device}")
        try:
            model_a, metadata_a = whisperx.load_align_model(
                language_code=language_code,
                device=device,
                model_dir="/app/model_cache/alignment" # Opcional
            )
            loaded_alignment_models[align_key] = (model_a, metadata_a)
            logger.info(f"Modelo de alineación para {language_code} cargado.")
        except Exception as e:
            logger.error(f"Error cargando modelo de alineación para {language_code}: {e}", exc_info=True)
            # Si no hay modelo de alineación para un idioma, whisperx puede lanzar un error específico
            if "No pre-trained wav2vec2 model found" in str(e) or "No such file or directory" in str(e):
                 raise HTTPException(status_code=400, detail=f"No hay modelo de alineación disponible para el idioma: {language_code}. Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo de alineación: {str(e)}")
    return loaded_alignment_models[align_key]

def get_diarization_pipeline(hf_token: str, device: str):
    diarize_key = (hf_token is not None, device)
    if diarize_key not in loaded_diarization_pipelines:
        if not hf_token:
            logger.warning("No se proporcionó HF_TOKEN. La diarización se omite.")
            loaded_diarization_pipelines[diarize_key] = None
            return None
        logger.info(f"Cargando pipeline de diarización en {device}")
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            # Mover a GPU solo si el dispositivo es CUDA. pyannote puede manejar 'cpu' también.
            if device == "cuda":
                pipeline.to(torch.device(device))
            loaded_diarization_pipelines[diarize_key] = pipeline
            logger.info("Pipeline de diarización cargada.")
        except ImportError:
            logger.error("pyannote.audio no está instalado. Por favor, instálalo para usar diarización.")
            loaded_diarization_pipelines[diarize_key] = None
            return None
        except Exception as e:
            logger.error(f"Error cargando pipeline de diarización: {e}", exc_info=True)
            loaded_diarization_pipelines[diarize_key] = None # Marcar como no disponible para evitar reintentos
            raise HTTPException(status_code=500, detail=f"Error al cargar pipeline de diarización: {str(e)}. Verifique su HF_TOKEN y la conexión a Hugging Face.")
    return loaded_diarization_pipelines[diarize_key]

# --- Helper function to clear models from memory ---
def _clear_models_from_memory():
    global loaded_models, loaded_alignment_models, loaded_diarization_pipelines, DEVICE, logger

    logger.info("Starting to clear all AI models from memory...")

    # Clear Whisper models
    if loaded_models:
        logger.info(f"Clearing {len(loaded_models)} Whisper model(s)...")
        # Iterate over a copy of values if direct deletion during iteration is problematic
        models_to_delete = list(loaded_models.values())
        for model in models_to_delete:
            try:
                del model
            except Exception as e:
                logger.error(f"Error deleting a Whisper model object: {e}", exc_info=True)
        loaded_models.clear()
        logger.info("Whisper models dictionary cleared.")

    # Clear Alignment models
    if loaded_alignment_models:
        logger.info(f"Clearing {len(loaded_alignment_models)} Alignment model(s)...")
        alignment_models_to_delete = list(loaded_alignment_models.values())
        for model_tuple in alignment_models_to_delete:
            if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
                model_a, metadata_a = model_tuple
                try:
                    del model_a
                except Exception as e:
                    logger.error(f"Error deleting alignment model_a: {e}", exc_info=True)
                try:
                    del metadata_a
                except Exception as e:
                    logger.error(f"Error deleting alignment metadata_a: {e}", exc_info=True)
            else:
                logger.warning(f"Unexpected item in loaded_alignment_models: {model_tuple}")
        loaded_alignment_models.clear()
        logger.info("Alignment models dictionary cleared.")

    # Clear Diarization pipelines
    if loaded_diarization_pipelines:
        logger.info(f"Clearing {len(loaded_diarization_pipelines)} Diarization pipeline(s)...")
        pipelines_to_delete = list(loaded_diarization_pipelines.values())
        for pipeline in pipelines_to_delete:
            if pipeline is not None: # Ensure pipeline is not None before attempting deletion
                try:
                    del pipeline
                except Exception as e:
                    logger.error(f"Error deleting a Diarization pipeline object: {e}", exc_info=True)
        loaded_diarization_pipelines.clear()
        logger.info("Diarization pipelines dictionary cleared.")

    if DEVICE == "cuda":
        logger.info("Clearing CUDA cache...")
        try:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        except Exception as e:
            logger.error(f"Error clearing CUDA cache: {e}", exc_info=True)
    
    logger.info("Model clearing process completed.")

# --- Endpoint de Transcripción ---
@app.post("/transcribe/")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model_name: str = Form("base"),
    language: str = Form(None),
    batch_size: int = Form(16), # WhisperX recomienda 4, 8, 16 para GPU. Ajustar según VRAM.
    align_audio: bool = Form(True),
    diarize_audio: bool = Form(False),
    hf_token: str = Form(None),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None)
):
    current_compute_type = COMPUTE_TYPE_GPU if DEVICE == "cuda" else COMPUTE_TYPE_CPU
    logger.info(f"Solicitud de transcripción: model={model_name}, lang={language or 'auto'}, align={align_audio}, diarize={diarize_audio}, batch_size={batch_size}")

    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{os.path.splitext(audio_file.filename)[1] if audio_file.filename else '.tmp'}")

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Archivo de audio guardado en: {temp_audio_path}")

        # 1. Cargar modelo Whisper
        # Pasar 'language' a get_model por si es 'large-v3' y se quiere una carga optimizada.
        model = get_model(model_name, DEVICE, current_compute_type, language=language if model_name == "large-v3" else None)


        # 2. Transcribir
        logger.info("Iniciando transcripción...")
        # El modelo transcribe directamente la ruta del archivo
        # Para large-v3, si no se especificó idioma en load_model, pasarlo aquí si está disponible
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs['language'] = language

        result = model.transcribe(temp_audio_path, batch_size=batch_size, print_progress=False, **transcribe_kwargs)
        detected_language = result["language"] # Este es el idioma detectado/usado por Whisper
        logger.info(f"Transcripción completada. Idioma detectado/usado: {detected_language}")

        loaded_audio_data = None # Inicializar variable para datos de audio cargados

        # 3. Alinear transcripción (opcional)
        alignment_performed_successfully = False
        if align_audio:
            if not detected_language:
                logger.warning("No se pudo detectar el idioma (o no se proporcionó y el modelo no lo devolvió), saltando la alineación.")
                result["warning_alignment"] = "No se pudo determinar el idioma para la alineación."
            else:
                try:
                    align_model, align_metadata = get_alignment_model(detected_language, DEVICE)
                    logger.info(f"Alineando transcripción para idioma: {detected_language}...")
                    if loaded_audio_data is None:
                        logger.info("Cargando datos de audio para alineación...")
                        loaded_audio_data = whisperx.load_audio(temp_audio_path)

                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        align_metadata,
                        loaded_audio_data, # Usar datos de audio cargados
                        DEVICE,
                        return_char_alignments=False
                    )
                    logger.info("Alineación completada.")
                    alignment_performed_successfully = True
                except HTTPException as e: # Errores específicos de carga de modelo de alineación
                    logger.warning(f"No se pudo realizar la alineación: {e.detail}")
                    result["warning_alignment"] = f"Alineación fallida: {e.detail}"
                except Exception as e:
                    logger.error(f"Error inesperado durante la alineación: {e}", exc_info=True)
                    result["warning_alignment"] = f"Error inesperado en alineación: {str(e)}"

        # 4. Diarizar (opcional)
        diarization_performed_successfully = False
        if diarize_audio:
            # La diarización requiere que la alineación haya ocurrido para asignar hablantes a palabras.
            # WhisperX `assign_word_speakers` espera que los segmentos tengan 'word' timings.
            if not align_audio or not alignment_performed_successfully:
                logger.warning("Diarización solicitada, pero la alineación no se realizó o falló. La diarización se omitirá o no tendrá información a nivel de palabra.")
                result["warning_diarization"] = "La diarización a nivel de palabra requiere una alineación exitosa."
            else:
                diarize_pipeline = get_diarization_pipeline(hf_token, DEVICE)
                if diarize_pipeline:
                    logger.info("Iniciando diarización...")
                    try:
                        if loaded_audio_data is None:
                            # Este caso no debería ocurrir si la alineación (que carga el audio) es un prerrequisito.
                            logger.error("Error crítico: loaded_audio_data es None antes de la diarización, pero la alineación debió cargarlo.")
                            raise HTTPException(status_code=500, detail="Error interno: datos de audio no cargados para diarización debido a un fallo previo en la carga para alineación.")

                        logger.info("Usando datos de audio cargados previamente para diarización.")
                        diarize_segments = diarize_pipeline(
                            {"waveform": torch.from_numpy(loaded_audio_data).unsqueeze(0), "sample_rate": 16000},
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                        # Asegurarse de que 'result' contenga segmentos con 'words'
                        if "segments" in result and result["segments"] and "words" in result["segments"][0]:
                            result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
                            result["segments"] = result_with_speakers["segments"] # Actualizar segmentos con la info de speaker
                            # assign_word_speakers puede añadir un top-level 'speaker_segments' si se quiere
                            # result["speaker_segments"] = diarize_segments # Opcional: devolver los segmentos puros de pyannote
                            logger.info("Diarización completada y asignada.")
                            diarization_performed_successfully = True
                        else:
                            logger.warning("Los segmentos de transcripción no contienen información de palabras (posible fallo de alineación). No se pueden asignar hablantes a palabras.")
                            result["warning_diarization"] = "Fallo de alineación impidió asignación de hablantes a palabras."

                    except HTTPException: # Si la pipeline de diarización falla al cargar (ya manejado en get_diarization_pipeline)
                        raise
                    except Exception as e:
                        logger.error(f"Error durante la diarización: {e}", exc_info=True)
                        result["warning_diarization"] = f"Diarización fallida: {str(e)}"
                else:
                    logger.warning("Diarización solicitada pero la pipeline no está disponible (token HF o instalación).")
                    result["warning_diarization"] = "Pipeline de diarización no disponible (verificar HF_TOKEN)."

        final_response = {
            "language": detected_language,
            "language_probability": result.get("language_probability"),
            "segments": result.get("segments", []), # Asegurar que siempre haya una lista
            "full_text": result.get("text", " ".join([s.get('text', '').strip() for s in result.get("segments", [])])),
            "options_used": {
                "model_name": model_name,
                "language_requested": language,
                "batch_size": batch_size,
                "alignment_performed": align_audio and alignment_performed_successfully,
                "diarization_performed": diarize_audio and diarization_performed_successfully,
                "device": DEVICE,
                "compute_type": current_compute_type
            }
        }
        if "warning_alignment" in result:
            final_response["options_used"]["alignment_warning"] = result["warning_alignment"]
        if "warning_diarization" in result:
            final_response["options_used"]["diarization_warning"] = result["warning_diarization"]


        return JSONResponse(content=final_response)

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error procesando la solicitud: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal {temp_dir} eliminado.")
        if audio_file:
            audio_file.file.close()
        
        # Aggressively clear models from memory after each request
        _clear_models_from_memory()

@app.get("/health")
async def health_check():
    return {"status": "ok", "device_available": DEVICE, "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}

# Para desarrollo local (no usado por Docker directamente)
if __name__ == "__main__":
    import uvicorn
    # Para probar diarización localmente con GPU, asegúrate de que CUDA y pyannote estén configurados
    # y exporta tu token de Hugging Face:
    # export HF_TOKEN="tu_token_aqui"
    #
    # Opcionalmente, para probar en CPU si no tienes GPU localmente:
    # DEVICE = "cpu" # Forzar CPU para prueba local
    # logger.info(f"Forzando CPU para desarrollo local.")

    uvicorn.run(app, host="0.0.0.0", port=8000)