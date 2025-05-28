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
logger.info(f"Máximo de modelos Whisper concurrentes permitidos en memoria: {MAX_WHISPER_MODELS if MAX_WHISPER_MODELS != float('inf') else 'Ilimitado'}")

loaded_models = collections.OrderedDict() # Usar OrderedDict para LRU

# Tipos de cómputo recomendados para GPU y CPU
COMPUTE_TYPE_GPU = "float16"
COMPUTE_TYPE_CPU = "int8"

# --- Funciones auxiliares ---

def get_model(model_name: str, device: str, compute_type: str, language: str = None):
    """
    Carga o recupera un modelo Whisper de la caché LRU.
    Utiliza las variables globales `loaded_models` y `MAX_WHISPER_MODELS`.
    """
    logger.info(f"Solicitando modelo Whisper: {model_name}, dispositivo: {device}, cómputo: {compute_type}, idioma: {language}")
    logger.debug(f"Estado actual de loaded_models: {list(loaded_models.keys())}")

    model_key_suffix = f"_lang-{language}" if model_name == "large-v3" and language else ""
    model_key = (model_name + model_key_suffix, device, compute_type)

    if model_key in loaded_models:
        logger.info(f"Modelo {model_key} encontrado en caché. Moviéndolo a 'más recientemente usado'.")
        # Mover al final (más recientemente usado)
        model = loaded_models.pop(model_key)
        loaded_models[model_key] = model
        return model
    else:
        logger.info(f"Modelo {model_key} no encontrado en caché. Se procederá a cargarlo.")

        # Comprobar si la caché está llena ANTES de cargar el nuevo modelo
        # y si MAX_WHISPER_MODELS no es infinito (lo que deshabilita la expulsión)
        if len(loaded_models) >= MAX_WHISPER_MODELS and MAX_WHISPER_MODELS != float('inf'):
            # Expulsar el modelo menos recientemente usado (primer item en OrderedDict)
            oldest_key, oldest_model = loaded_models.popitem(last=False) # LRU item
            logger.info(f"Caché llena (max_models={MAX_WHISPER_MODELS}). Expulsando modelo LRU: {oldest_key} para liberar memoria.")
            try:
                del oldest_model
                if device == "cuda": # Solo vaciar caché CUDA si el dispositivo actual es CUDA
                    torch.cuda.empty_cache()
                logger.info(f"Modelo {oldest_key} expulsado. Caché de CUDA vaciada (si aplica).")
            except Exception as e:
                logger.error(f"Error durante la expulsión del modelo {oldest_key}: {e}", exc_info=True)
        
        logger.info(f"Cargando nuevo modelo Whisper: {model_key[0]} (idioma: {'especificado ' + language if model_key_suffix else 'no especificado'}) en {device} con compute_type {compute_type}")
        try:
            model_kwargs = {}
            if model_name == "large-v3" and language:
                model_kwargs['language'] = language

            download_root_path = "/app/model_cache/whisper"
            
            new_model = whisperx.load_model(
                model_name,
                device,
                compute_type=compute_type,
                download_root=download_root_path,
                **model_kwargs
            )
            loaded_models[model_key] = new_model # Añadir a la caché
            logger.info(f"Modelo {model_key} cargado y añadido a la caché. Tamaño actual de la caché: {len(loaded_models)}/{MAX_WHISPER_MODELS if MAX_WHISPER_MODELS != float('inf') else 'ilimitado'}.")
            return new_model
        except Exception as e:
            logger.error(f"Error cargando modelo Whisper {model_key}: {e}", exc_info=True)
            if model_key in loaded_models:
                 del loaded_models[model_key]
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo Whisper {model_key}: {str(e)}")

def get_alignment_model(language_code: str, device: str):
    align_key = (language_code, device)
    if align_key not in loaded_alignment_models:
        logger.info(f"Cargando modelo de alineación para idioma: {language_code} en {device}")
        try:
            model_a, metadata_a = whisperx.load_align_model(
                language_code=language_code,
                device=device,
                model_dir="/app/model_cache/alignment"
            )
            loaded_alignment_models[align_key] = (model_a, metadata_a)
            logger.info(f"Modelo de alineación para {language_code} cargado.")
        except Exception as e:
            logger.error(f"Error cargando modelo de alineación para {language_code}: {e}", exc_info=True)
            if "No pre-trained wav2vec2 model found" in str(e) or "No such file or directory" in str(e):
                 raise HTTPException(status_code=400, detail=f"No hay modelo de alineación disponible para el idioma: {language_code}. Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo de alineación: {str(e)}")
    else:
        logger.info(f"Modelo de alineación para {language_code} encontrado en caché.")
    return loaded_alignment_models[align_key]

def get_diarization_pipeline(hf_token_from_request: str, device: str):
    # Usar el token de la solicitud si se proporciona, si no, el de la variable de entorno
    effective_hf_token = hf_token_from_request or os.getenv("HF_TOKEN")

    diarize_key = (effective_hf_token is not None, device) # La clave depende de si hay un token y del dispositivo
    
    if diarize_key not in loaded_diarization_pipelines:
        if not effective_hf_token:
            logger.warning("No se proporcionó HF_TOKEN (ni en la solicitud ni en el entorno). La diarización se omite.")
            loaded_diarization_pipelines[diarize_key] = None # Cachear que no hay pipeline para esta config
            return None
        
        logger.info(f"Cargando pipeline de diarización en {device} (usando token provisto).")
        try:
            from pyannote.audio import Pipeline # Importar aquí para evitar error si no se usa diarización
            
            # Especificar el directorio de caché para los modelos de Hugging Face
            # HF_HOME o XDG_CACHE_HOME deben estar configurados en el Dockerfile para apuntar a /app/model_cache/huggingface
            # Si no, pyannote usará ~/.cache/huggingface por defecto.
            # Pipeline.from_pretrained ya usa la caché de HF.
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=effective_hf_token
                # cache_dir="/app/model_cache/huggingface" # Opcional, si HF_HOME no está configurado globalmente
            )

            if device == "cuda":
                pipeline.to(torch.device(device))
            loaded_diarization_pipelines[diarize_key] = pipeline
            logger.info("Pipeline de diarización cargada.")
        except ImportError:
            logger.error("pyannote.audio no está instalado. Por favor, instálalo para usar diarización.")
            loaded_diarization_pipelines[diarize_key] = None
            # No lanzar HTTPException aquí, permitir que la transcripción continúe sin diarización
            return None
        except Exception as e:
            logger.error(f"Error cargando pipeline de diarización: {e}", exc_info=True)
            loaded_diarization_pipelines[diarize_key] = None
            # Considerar si lanzar HTTPException o permitir continuar sin diarización
            # Por ahora, permitimos continuar, pero el usuario recibirá una advertencia.
            # raise HTTPException(status_code=500, detail=f"Error al cargar pipeline de diarización: {str(e)}. Verifique su HF_TOKEN y la conexión a Hugging Face.")
            return None # Indica que la carga falló
    else:
        logger.info("Pipeline de diarización encontrada en caché.")
        if loaded_diarization_pipelines[diarize_key] is None and effective_hf_token:
            logger.warning("Intento previo de cargar pipeline de diarización (con token) falló. Usando resultado cacheado (None).")
        elif loaded_diarization_pipelines[diarize_key] is None and not effective_hf_token:
             logger.info("Pipeline de diarización no disponible (sin token). Usando resultado cacheado (None).")


    return loaded_diarization_pipelines[diarize_key]

# --- Endpoint de Transcripción ---
@app.post("/transcribe/")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model_name: str = Form("base"),
    language: str = Form(None),
    batch_size: int = Form(16),
    align_audio: bool = Form(True),
    diarize_audio: bool = Form(False),
    hf_token: str = Form(None), # Token de la solicitud
    min_speakers: int = Form(None),
    max_speakers: int = Form(None)
):
    current_compute_type = COMPUTE_TYPE_GPU if DEVICE == "cuda" else COMPUTE_TYPE_CPU
    request_id = str(uuid.uuid4())
    logger.info(f"ID Solicitud [{request_id}]: model={model_name}, lang={language or 'auto'}, align={align_audio}, diarize={diarize_audio}, batch_size={batch_size}")

    temp_dir = None # Inicializar fuera del try para el finally
    try:
        temp_dir = tempfile.mkdtemp(prefix=f"req_{request_id}_")
        # Usar un nombre de archivo único dentro del directorio temporal
        original_filename = audio_file.filename if audio_file.filename else "audiofile"
        # Limpiar el nombre del archivo para evitar problemas con caracteres especiales
        safe_filename_base = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in os.path.splitext(original_filename)[0])
        safe_extension = "".join(c if c.isalnum() or c == '.' else '_' for c in os.path.splitext(original_filename)[1])
        temp_audio_path = os.path.join(temp_dir, f"{safe_filename_base}{safe_extension}")


        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"ID Solicitud [{request_id}]: Archivo de audio guardado en: {temp_audio_path}")

        # 1. Cargar modelo Whisper
        model = get_model(model_name, DEVICE, current_compute_type, language=language if model_name == "large-v3" else None)

        # 2. Transcribir
        logger.info(f"ID Solicitud [{request_id}]: Iniciando transcripción...")
        transcribe_kwargs = {}
        if language: # Solo pasar 'language' a transcribe si se especificó
            transcribe_kwargs['language'] = language
        
        result = model.transcribe(temp_audio_path, batch_size=batch_size, print_progress=False, **transcribe_kwargs)
        detected_language = result["language"]
        logger.info(f"ID Solicitud [{request_id}]: Transcripción completada. Idioma detectado/usado: {detected_language}")

        loaded_audio_data = None

        # 3. Alinear transcripción (opcional)
        alignment_performed_successfully = False
        if align_audio:
            if not detected_language:
                logger.warning(f"ID Solicitud [{request_id}]: No se pudo detectar el idioma, saltando la alineación.")
                result["warning_alignment"] = "No se pudo determinar el idioma para la alineación."
            else:
                try:
                    align_model, align_metadata = get_alignment_model(detected_language, DEVICE)
                    logger.info(f"ID Solicitud [{request_id}]: Alineando transcripción para idioma: {detected_language}...")
                    if loaded_audio_data is None:
                        logger.info(f"ID Solicitud [{request_id}]: Cargando datos de audio para alineación...")
                        loaded_audio_data = whisperx.load_audio(temp_audio_path)
                    
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        align_metadata,
                        loaded_audio_data,
                        DEVICE,
                        return_char_alignments=False
                    )
                    logger.info(f"ID Solicitud [{request_id}]: Alineación completada.")
                    alignment_performed_successfully = True
                except HTTPException as e:
                    logger.warning(f"ID Solicitud [{request_id}]: No se pudo realizar la alineación: {e.detail}")
                    result["warning_alignment"] = f"Alineación fallida: {e.detail}"
                except Exception as e:
                    logger.error(f"ID Solicitud [{request_id}]: Error inesperado durante la alineación: {e}", exc_info=True)
                    result["warning_alignment"] = f"Error inesperado en alineación: {str(e)}"

        # 4. Diarizar (opcional)
        diarization_performed_successfully = False
        if diarize_audio:
            if not align_audio or not alignment_performed_successfully:
                logger.warning(f"ID Solicitud [{request_id}]: Diarización solicitada, pero la alineación no se realizó o falló. Se omitirá la asignación de hablantes a palabras.")
                result["warning_diarization"] = "La diarización a nivel de palabra requiere una alineación exitosa."
            
            # Intentar obtener el pipeline incluso si la alineación falló, podría ser útil para segmentos de hablante sin palabras
            diarize_pipeline = get_diarization_pipeline(hf_token, DEVICE) # Pasar token de la solicitud
            if diarize_pipeline:
                logger.info(f"ID Solicitud [{request_id}]: Iniciando diarización...")
                try:
                    if loaded_audio_data is None: # Cargar audio si no se hizo para alineación
                        logger.info(f"ID Solicitud [{request_id}]: Cargando datos de audio para diarización...")
                        loaded_audio_data = whisperx.load_audio(temp_audio_path)
                    
                    # Asegurarse de que loaded_audio_data es un tensor numpy para pyannote
                    # y convertirlo a tensor de PyTorch en el formato esperado por pyannote
                    # Pyannote espera una forma de onda como [1, N] (batch_size=1, num_samples)
                    waveform_tensor = torch.from_numpy(loaded_audio_data).unsqueeze(0)
                    audio_for_diarization = {"waveform": waveform_tensor, "sample_rate": 16000} # WhisperX siempre resamplea a 16kHz

                    diarize_segments = diarize_pipeline(
                        audio_for_diarization,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    
                    # Solo asignar a palabras si la alineación fue exitosa
                    if align_audio and alignment_performed_successfully:
                        if "segments" in result and result["segments"] and "words" in result["segments"][0]:
                            result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
                            result["segments"] = result_with_speakers["segments"]
                            logger.info(f"ID Solicitud [{request_id}]: Diarización completada y asignada a palabras.")
                            diarization_performed_successfully = True
                        else:
                            logger.warning(f"ID Solicitud [{request_id}]: Los segmentos no contienen 'words' post-alineación. No se pueden asignar hablantes a palabras.")
                            result["warning_diarization"] = (result.get("warning_diarization", "") + " Fallo de alineación impidió asignación de hablantes a palabras.").strip()
                            result["speaker_segments_raw"] = [{"start": turn.start, "end": turn.end, "speaker": label} for turn, _, label in diarize_segments.itertracks(yield_label=True)]
                    else:
                        # Si la alineación no se hizo o falló, al menos devolver los segmentos de diarización crudos
                        logger.info(f"ID Solicitud [{request_id}]: Diarización completada (segmentos de hablante). Asignación a palabras omitida.")
                        result["speaker_segments_raw"] = [{"start": turn.start, "end": turn.end, "speaker": label} for turn, _, label in diarize_segments.itertracks(yield_label=True)]
                        diarization_performed_successfully = True # La diarización en sí se realizó

                except HTTPException: # De get_diarization_pipeline si falla la carga del modelo
                    raise
                except Exception as e:
                    logger.error(f"ID Solicitud [{request_id}]: Error durante la diarización: {e}", exc_info=True)
                    result["warning_diarization"] = (result.get("warning_diarization", "") + f" Diarización fallida: {str(e)}").strip()
            else:
                logger.warning(f"ID Solicitud [{request_id}]: Diarización solicitada pero la pipeline no está disponible (verificar HF_TOKEN o instalación de pyannote).")
                result["warning_diarization"] = (result.get("warning_diarization", "") + " Pipeline de diarización no disponible.").strip()

        final_response = {
            "request_id": request_id,
            "language": detected_language,
            "language_probability": result.get("language_probability"),
            "segments": result.get("segments", []),
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
        if "speaker_segments_raw" in result:
             final_response["speaker_segments_raw"] = result["speaker_segments_raw"]


        return JSONResponse(content=final_response)

    except HTTPException as e:
        logger.error(f"ID Solicitud [{request_id}]: HTTP Exception: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"ID Solicitud [{request_id}]: Error procesando la solicitud: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor [{request_id}]: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"ID Solicitud [{request_id}]: Directorio temporal {temp_dir} eliminado.")
        if audio_file:
            try:
                audio_file.file.close()
            except Exception as e:
                logger.warning(f"ID Solicitud [{request_id}]: Error cerrando archivo de audio: {e}")
        
        # NO LLAMAR A _clear_models_from_memory() aquí para que la caché persista entre solicitudes.

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "device_available": DEVICE, 
        "torch_version": torch.__version__, 
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" and torch.cuda.is_available() else "N/A",
        "max_whisper_models_in_cache": MAX_WHISPER_MODELS if MAX_WHISPER_MODELS != float('inf') else "Unlimited",
        "current_whisper_models_in_cache": len(loaded_models),
        "cached_whisper_model_keys": list(loaded_models.keys()),
        "cached_alignment_model_keys": list(loaded_alignment_models.keys()),
        "cached_diarization_pipeline_keys": list(loaded_diarization_pipelines.keys())
    }

# Para desarrollo local (no usado por Docker directamente)
if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor Uvicorn para desarrollo local...")
    # Para probar diarización localmente con GPU, asegúrate de que CUDA y pyannote estén configurados
    # y exporta tu token de Hugging Face si es necesario:
    # export HF_TOKEN="tu_token_aqui"
    
    # Ejemplo para forzar CPU si no tienes GPU localmente o para pruebas:
    # DEVICE = "cpu"
    # logger.info(f"Forzando CPU para desarrollo local. Dispositivo actual: {DEVICE}")

    uvicorn.run(app, host="0.0.0.0", port=8000)