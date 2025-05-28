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
from pyannote.audio import Pipeline # Moved import to top for clarity
from pyannote.core import Segment as PyannoteSegment # For type hinting if needed

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhisperX API (GPU)",
    description="API para transcripción de audio usando WhisperX con alineación y diarización, optimizada para GPU.",
    version="0.1.1" # incremented version
)

# Variables globales para los modelos
loaded_alignment_models = {}
loaded_diarization_pipelines = {} # diarization pipeline (singular, as it's not language-specific)

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

MAX_WHISPER_MODELS = int(os.environ.get("MAX_WHISPER_MODELS", "1"))
if MAX_WHISPER_MODELS <= 0:
    MAX_WHISPER_MODELS = float('inf')
logger.info(f"Maximum concurrent Whisper models allowed in memory: {MAX_WHISPER_MODELS if MAX_WHISPER_MODELS != float('inf') else 'Unlimited'}")

loaded_models = collections.OrderedDict()

COMPUTE_TYPE_GPU = "float16"
COMPUTE_TYPE_CPU = "int8"

# --- Funciones auxiliares ---
def get_model(model_name: str, device: str, compute_type: str, language: str = None):
    model_key_suffix = f"_lang-{language}" if model_name == "large-v3" and language else ""
    model_key = (model_name + model_key_suffix, device, compute_type)

    if model_key in loaded_models:
        logger.info(f"Model {model_key} found in cache. Moving to end (most recently used).")
        loaded_models.move_to_end(model_key)
        return loaded_models[model_key]

    logger.info(f"Model {model_key} not found in cache. Attempting to load.")

    if len(loaded_models) >= MAX_WHISPER_MODELS:
        oldest_key, oldest_model = loaded_models.popitem(last=False)
        logger.info(f"Cache is full (max_models={MAX_WHISPER_MODELS}). Evicting model {oldest_key} to free up memory.")
        try:
            del oldest_model
            if device == "cuda": # Changed from DEVICE to device parameter
                torch.cuda.empty_cache()
            logger.info(f"Model {oldest_key} evicted and CUDA cache cleared (if on CUDA).")
        except Exception as e:
            logger.error(f"Error during eviction of model {oldest_key}: {e}", exc_info=True)

    logger.info(f"Loading Whisper model: {model_name} ({model_key_suffix.replace('_lang-','') if model_key_suffix else 'no lang specified'}) on {device} with compute_type {compute_type}")
    try:
        model_kwargs = {}
        if model_name == "large-v3" and language:
            model_kwargs['language'] = language

        download_root_path = "/app/model_cache/whisper"
        # os.makedirs(download_root_path, exist_ok=True) # Dockerfile handles this

        new_model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            download_root=download_root_path,
            **model_kwargs
        )
        loaded_models[model_key] = new_model
        logger.info(f"Model {model_key} loaded and added to cache. Current cache size: {len(loaded_models)}.")
        return new_model
    except Exception as e:
        logger.error(f"Error loading Whisper model {model_key}: {e}", exc_info=True)
        if model_key in loaded_models:
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
                model_dir="/app/model_cache/alignment"
            )
            loaded_alignment_models[align_key] = (model_a, metadata_a)
            logger.info(f"Modelo de alineación para {language_code} cargado.")
        except Exception as e:
            logger.error(f"Error cargando modelo de alineación para {language_code}: {e}", exc_info=True)
            if "No pre-trained wav2vec2 model found" in str(e) or "No such file or directory" in str(e):
                 raise HTTPException(status_code=400, detail=f"No hay modelo de alineación disponible para el idioma: {language_code}. Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo de alineación: {str(e)}")
    return loaded_alignment_models[align_key]

def get_diarization_pipeline(hf_token: str, device: str):
    # Diarization pipeline is not token-specific for loading, but token is for usage.
    # Key just on device for simplicity, assuming one pyannote pipeline.
    diarize_key = device # Simplified key
    if diarize_key not in loaded_diarization_pipelines:
        if not hf_token: # Token is still needed for the first time download/use from_pretrained
            logger.warning("No se proporcionó HF_TOKEN. La diarización no se cargará/utilizará si el modelo no está cacheado localmente.")
            # We can't definitively say it won't work if model is cached, so don't return None yet.
            # Let from_pretrained handle it.
            # loaded_diarization_pipelines[diarize_key] = None # Don't do this here
            # return None
        logger.info(f"Cargando pipeline de diarización en {device}")
        try:
            # Pipeline import moved to top
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            if device == "cuda": # pyannote handles 'cpu' string directly if passed to .to()
                pipeline.to(torch.device(device))
            loaded_diarization_pipelines[diarize_key] = pipeline
            logger.info("Pipeline de diarización cargada.")
        except ImportError:
            logger.error("pyannote.audio no está instalado. Por favor, instálalo para usar diarización.")
            loaded_diarization_pipelines[diarize_key] = None # Mark as unavailable
            return None
        except Exception as e: # This can include huggingface_hub.utils._errors.HfHubHTTPError for bad token
            logger.error(f"Error cargando pipeline de diarización: {e}", exc_info=True)
            loaded_diarization_pipelines[diarize_key] = None # Mark as unavailable
            # Raise specific error if it's a token/auth issue, otherwise generic
            if "401 Client Error" in str(e) or "Unauthorized" in str(e) or "authentication" in str(e).lower():
                 raise HTTPException(status_code=401, detail=f"Error de autenticación al cargar pipeline de diarización: {str(e)}. Verifique su HF_TOKEN.")
            raise HTTPException(status_code=500, detail=f"Error al cargar pipeline de diarización: {str(e)}.")
    
    # If pipeline is None because hf_token was not provided at load time, but it's in cache
    # it might still work. However, pyannote usually requires token for actual processing too.
    # If it's marked as None due to previous load failure, return None.
    if loaded_diarization_pipelines.get(diarize_key) is None and diarize_key in loaded_diarization_pipelines:
        logger.warning("Pipeline de diarización no pudo ser cargada previamente.")
        return None
        
    return loaded_diarization_pipelines.get(diarize_key)


# --- Endpoint de Transcripción ---
@app.post("/transcribe/")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model_name: str = Form("base"),
    language: str = Form(None),
    batch_size: int = Form(16),
    align_audio: bool = Form(True),
    diarize_audio: bool = Form(False),
    hf_token: str = Form(None), # Now consistently used by get_diarization_pipeline
    min_speakers: int = Form(None),
    max_speakers: int = Form(None),
    word_level_timestamps: bool = Form(True)
):
    current_compute_type = COMPUTE_TYPE_GPU if DEVICE == "cuda" else COMPUTE_TYPE_CPU
    logger.info(f"Solicitud de transcripción: model={model_name}, lang={language or 'auto'}, align={align_audio}, diarize={diarize_audio}, batch_size={batch_size}, word_level_timestamps={word_level_timestamps}")

    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{os.path.splitext(audio_file.filename)[1] if audio_file.filename else '.tmp'}")

    # Initialize response structure elements
    response_warnings = {}
    raw_speaker_activity = None # To store pyannote's raw output

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Archivo de audio guardado en: {temp_audio_path}")

        model = get_model(model_name, DEVICE, current_compute_type, language=language if model_name == "large-v3" else None)

        transcribe_kwargs = {}
        if language:
            transcribe_kwargs['language'] = language
        result = model.transcribe(temp_audio_path, batch_size=batch_size, print_progress=False, **transcribe_kwargs)
        detected_language = result["language"]
        logger.info(f"Transcripción completada. Idioma detectado/usado: {detected_language}")

        loaded_audio_data = None # Initialize

        # 3. Alinear transcripción (opcional, controlled by word_level_timestamps)
        alignment_performed_for_words = False
        if align_audio and word_level_timestamps: # Alignment is for word-level timestamps
            if not detected_language:
                logger.warning("No se pudo detectar el idioma, saltando la alineación para marcas de tiempo de palabras.")
                response_warnings["alignment"] = "No se pudo determinar el idioma para la alineación de palabras."
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
                        loaded_audio_data,
                        DEVICE,
                        return_char_alignments=False
                    )
                    logger.info("Alineación para marcas de tiempo de palabras completada.")
                    alignment_performed_for_words = True
                except HTTPException as e:
                    logger.warning(f"No se pudo realizar la alineación para marcas de tiempo de palabras: {e.detail}")
                    response_warnings["alignment"] = f"Alineación de palabras fallida: {e.detail}"
                except Exception as e:
                    logger.error(f"Error inesperado durante la alineación de palabras: {e}", exc_info=True)
                    response_warnings["alignment"] = f"Error inesperado en alineación de palabras: {str(e)}"
        elif align_audio and not word_level_timestamps:
            logger.info("Alineación solicitada (`align_audio=True`) pero `word_level_timestamps=False`. No se realizará alineación para marcas de tiempo de palabras.")
            response_warnings["alignment"] = "Alineación para marcas de tiempo de palabras no realizada ya que `word_level_timestamps` es `False`."
        elif not align_audio:
            logger.info("Alineación no solicitada (`align_audio=False`).")


        # 4. Diarizar (opcional)
        word_level_diarization_assigned = False
        speaker_turns_generated = False

        if diarize_audio:
            # Try to get/load diarization pipeline first
            diarize_pipeline = None
            try:
                diarize_pipeline = get_diarization_pipeline(hf_token, DEVICE)
            except HTTPException as e: # Catch errors from get_diarization_pipeline (e.g. bad token, load fail)
                logger.warning(f"No se pudo cargar/obtener la pipeline de diarización: {e.detail}")
                response_warnings["diarization"] = f"Pipeline de diarización no disponible: {e.detail}"
            
            if diarize_pipeline:
                logger.info("Iniciando diarización de turnos de hablante...")
                try:
                    if loaded_audio_data is None:
                        logger.info("Cargando datos de audio para diarización...")
                        loaded_audio_data = whisperx.load_audio(temp_audio_path)
                    
                    # Ensure audio data is a Tensor on the correct device for Pyannote
                    # Pyannote expects [batch_size, num_channels, num_samples] or [batch_size, num_samples]
                    # whisperx.load_audio returns a NumPy array (mono, 16kHz)
                    audio_tensor = torch.from_numpy(loaded_audio_data).float() # Ensure float
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0) # Add batch dimension: [1, num_samples]
                    
                    # Pyannote pipeline expects a dict with 'waveform' and 'sample_rate'
                    # waveform should be [batch, channels, samples] or [samples] if pipeline handles unsqueeze
                    # For diarization-3.1, it seems to work with [1, num_samples] for mono.
                    pyannote_input = {"waveform": audio_tensor, "sample_rate": 16000}

                    diarize_segments_pyannote = diarize_pipeline(
                        pyannote_input, # Pass the dictionary
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        # num_speakers=num_speakers # if you prefer fixed number
                    )
                    
                    # Store raw pyannote segments (converted to a simpler list of dicts)
                    raw_speaker_activity = []
                    for turn, _, speaker in diarize_segments_pyannote.itertracks(yield_label=True):
                        raw_speaker_activity.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    speaker_turns_generated = True
                    logger.info("Diarización de turnos de hablante completada.")

                    # Now, attempt to assign speakers to words IF word-level alignment was successful
                    if alignment_performed_for_words:
                        if "segments" in result and result["segments"] and "words" in result["segments"][0]:
                            try:
                                logger.info("Asignando hablantes a palabras...")
                                result_with_speakers = whisperx.assign_word_speakers(diarize_segments_pyannote, result)
                                result["segments"] = result_with_speakers["segments"]
                                word_level_diarization_assigned = True
                                logger.info("Asignación de hablantes a palabras completada.")
                            except KeyError as e:
                                msg = f"Fallo en asignación de hablantes a palabras (KeyError: {e}). Puede ser por etiquetas de hablante inesperadas."
                                logger.warning(msg + f" Segmentos de Pyannote: {diarize_segments_pyannote}")
                                response_warnings["diarization_word_assignment"] = msg
                            except Exception as e_assign:
                                msg = f"Error inesperado durante la asignación de hablantes a palabras: {str(e_assign)}"
                                logger.error(msg, exc_info=True)
                                response_warnings["diarization_word_assignment"] = msg
                        else:
                            msg = "Los segmentos de transcripción no contienen información de palabras (posible fallo de alineación). No se pueden asignar hablantes a palabras."
                            logger.warning(msg)
                            response_warnings["diarization_word_assignment"] = msg
                    else:
                        msg = "La asignación de hablantes a palabras se omitió porque la alineación para marcas de tiempo de palabras no se realizó o falló."
                        logger.info(msg)
                        if "diarization" not in response_warnings : # Avoid overwriting more specific diarization pipeline load errors
                             response_warnings["diarization"] = msg # General info if no other diarization error

                except HTTPException: # Should be caught by the get_pipeline_diarization call
                    raise
                except Exception as e:
                    msg = f"Error durante la diarización de turnos de hablante: {str(e)}"
                    logger.error(msg, exc_info=True)
                    response_warnings["diarization"] = msg # Overwrites previous generic if more specific error here
            # else: diarize_pipeline was None (handled by response_warnings["diarization"] already)

        elif not diarize_audio:
            logger.info("Diarización no solicitada.")


        final_response = {
            "language": detected_language,
            "language_probability": result.get("language_probability"),
            "segments": result.get("segments", []),
            "full_text": result.get("text", " ".join([s.get('text', '').strip() for s in result.get("segments", [])])),
            "options_used": {
                "model_name": model_name,
                "language_requested": language,
                "batch_size": batch_size,
                "alignment_requested_for_words": align_audio and word_level_timestamps,
                "alignment_performed_for_words": alignment_performed_for_words,
                "diarization_requested": diarize_audio,
                "diarization_speaker_turns_generated": speaker_turns_generated,
                "diarization_word_level_assigned": word_level_diarization_assigned,
                "word_level_timestamps_requested": word_level_timestamps, # This controls if alignment for words is attempted
                "device": DEVICE,
                "compute_type": current_compute_type
            }
        }
        if raw_speaker_activity:
            final_response["speaker_activity_segments"] = raw_speaker_activity
        
        if response_warnings:
            final_response["warnings"] = response_warnings

        return JSONResponse(content=final_response)

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        # Ensure temp dir is cleaned up even for expected HTTP exceptions
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal {temp_dir} eliminado tras HTTPException.")
        if audio_file:
            audio_file.file.close()
        raise e # Re-raise the HTTPException
    except Exception as e:
        logger.error(f"Error procesando la solicitud: {e}", exc_info=True)
        # For unexpected errors, also ensure cleanup and then raise a generic 500
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal {temp_dir} eliminado tras Exception.")
        if audio_file:
            audio_file.file.close()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    finally:
        # This finally block might run after an exception has already cleaned up.
        # Check existence before attempting to remove again.
        if 'temp_dir' in locals() and os.path.exists(temp_dir): # Ensure temp_dir was defined
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal {temp_dir} eliminado (bloque finally).")
        if audio_file and not audio_file.file.closed: # Ensure file was passed and not already closed
            audio_file.file.close()


@app.get("/health")
async def health_check():
    # Simple check: is CUDA available as expected?
    cuda_ok = True
    if DEVICE == "cuda" and not torch.cuda.is_available():
        cuda_ok = False
        logger.error("Health check: DEVICE is 'cuda' but torch.cuda.is_available() is False!")

    # Check loaded models (optional, could be verbose)
    # whisper_models_count = len(loaded_models)
    # alignment_models_count = len(loaded_alignment_models)
    # diarization_pipelines_count = len(loaded_diarization_pipelines)

    return {
        "status": "ok" if cuda_ok else "degraded",
        "device_in_use": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available_torch": torch.cuda.is_available(),
        "cuda_consistency_ok": cuda_ok,
        # "cached_whisper_models": whisper_models_count,
        # "cached_alignment_models": alignment_models_count,
        # "cached_diarization_pipelines": diarization_pipelines_count,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)