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
from pyannote.audio import Pipeline
from pyannote.core import Annotation # Para el type hint y el isinstance
import traceback # For detailed error logging

# Configuración de logging
logging.basicConfig(level=logging.DEBUG) # O logging.DEBUG para más detalle
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhisperX API (GPU)",
    description="API para transcripción de audio usando WhisperX con alineación y diarización, optimizada para GPU.",
    version="0.1.5" # Integrando parches en la base 0.1.1
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
            if device == "cuda":
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
    diarize_key = device
    if diarize_key not in loaded_diarization_pipelines:
        if not hf_token:
            logger.warning("No se proporcionó HF_TOKEN. La diarización no se cargará/utilizará si el modelo no está cacheado localmente.")
        logger.info(f"Cargando pipeline de diarización en {device}")
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
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
            loaded_diarization_pipelines[diarize_key] = None
            if "401 Client Error" in str(e) or "Unauthorized" in str(e) or "authentication" in str(e).lower():
                 raise HTTPException(status_code=401, detail=f"Error de autenticación al cargar pipeline de diarización: {str(e)}. Verifique su HF_TOKEN.")
            raise HTTPException(status_code=500, detail=f"Error al cargar pipeline de diarización: {str(e)}.")
    
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
    hf_token: str = Form(None),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None),
    word_level_timestamps: bool = Form(True)
):
    current_compute_type = COMPUTE_TYPE_GPU if DEVICE == "cuda" else COMPUTE_TYPE_CPU
    logger.info(f"Solicitud de transcripción: model={model_name}, lang={language or 'auto'}, align={align_audio}, diarize={diarize_audio}, batch_size={batch_size}, word_level_timestamps={word_level_timestamps}")

    temp_dir = tempfile.mkdtemp()
    original_filename = audio_file.filename if audio_file.filename else "audiofile"
    safe_original_filename = "".join(c for c in original_filename if c.isalnum() or c in ['.', '_', '-']).strip()
    if not safe_original_filename: safe_original_filename = "audio.tmp"
    temp_audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{safe_original_filename}")

    response_warnings = {}
    raw_speaker_activity = None
    detected_language = None

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Archivo de audio guardado en: {temp_audio_path}")

        model = get_model(model_name, DEVICE, current_compute_type, language=language if model_name == "large-v3" else None)

        transcribe_kwargs = {}
        if language:
            transcribe_kwargs['language'] = language
        
        result: dict = model.transcribe(temp_audio_path, batch_size=batch_size, print_progress=False, **transcribe_kwargs)
        
        if not isinstance(result, dict) or "language" not in result or "segments" not in result:
            logger.error(f"Resultado inesperado de model.transcribe. Tipo: {type(result)}. Valor: {str(result)[:500]}")
            raise HTTPException(status_code=500, detail="Error interno: la transcripción inicial no devolvió la estructura esperada.")
        
        detected_language = result.get("language")
        logger.info(f"Transcripción inicial completada. Idioma detectado/usado: {detected_language}")

        loaded_audio_data = None
        alignment_performed_for_words = False

        if align_audio and word_level_timestamps:
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
                    
                    aligned_segments = whisperx.align( # whisperx.align devuelve una LISTA de segmentos
                        result["segments"], 
                        align_model,
                        align_metadata,
                        loaded_audio_data,
                        DEVICE,
                        return_char_alignments=False
                    )
                    result["segments"] = aligned_segments # Actualizar los segmentos en el dict 'result'
                    
                    if aligned_segments and isinstance(aligned_segments, list) and len(aligned_segments) > 0 and \
                       isinstance(aligned_segments[0], dict) and "words" in aligned_segments[0]:
                        alignment_performed_for_words = True
                        logger.info("Alineación para marcas de tiempo de palabras completada.")
                    else:
                        logger.warning("Alineación realizada, pero los segmentos no parecen tener información de palabras.")
                        response_warnings["alignment"] = "Alineación realizada, pero sin información de palabras detectada."
                        
                except HTTPException as e:
                    logger.warning(f"No se pudo realizar la alineación para marcas de tiempo de palabras: {e.detail}")
                    response_warnings["alignment"] = f"Alineación de palabras fallida: {e.detail}"
                except Exception as e:
                    logger.error(f"Error inesperado durante la alineación de palabras: {e}", exc_info=True)
                    response_warnings["alignment"] = f"Error inesperado en alineación de palabras: {str(e)}"
        elif align_audio and not word_level_timestamps:
            logger.info("Alineación solicitada (`align_audio=True`) pero `word_level_timestamps=False`. No se realizará alineación para marcas de tiempo de palabras.")
        elif not align_audio:
            logger.info("Alineación no solicitada (`align_audio=False`).")

        if not isinstance(result, dict): # Verificación defensiva
            logger.error(f"CRITICAL: 'result' no es un diccionario después de la sección de alineación. Tipo: {type(result)}. Forzando a estructura básica.")
            result = {"segments": [], "language": detected_language if detected_language else None, "text": ""}
            response_warnings["internal_error_structure"] = "Error de estructura interna (post-alineación), resultados pueden ser incompletos."

        word_level_diarization_assigned = False
        speaker_turns_generated = False

        if diarize_audio:
            diarize_pipeline = None
            try:
                diarize_pipeline = get_diarization_pipeline(hf_token, DEVICE)
            except HTTPException as e:
                logger.warning(f"No se pudo cargar/obtener la pipeline de diarización: {e.detail}")
                response_warnings["diarization"] = f"Pipeline de diarización no disponible: {e.detail}"
            
            if diarize_pipeline:
                logger.info("Iniciando diarización de turnos de hablante...")
                try:
                    if loaded_audio_data is None:
                        logger.info("Cargando datos de audio para diarización...")
                        loaded_audio_data = whisperx.load_audio(temp_audio_path)
                    
                    audio_tensor = torch.from_numpy(loaded_audio_data).float()
                    if audio_tensor.ndim == 1: audio_tensor = audio_tensor.unsqueeze(0)
                    pyannote_input = {"waveform": audio_tensor, "sample_rate": 16000}

                    diarize_segments_pyannote: Annotation = diarize_pipeline(pyannote_input, min_speakers=min_speakers, max_speakers=max_speakers)
                    
                    raw_speaker_activity = [{"start": turn.start, "end": turn.end, "speaker": speaker} for turn, _, speaker in diarize_segments_pyannote.itertracks(yield_label=True)]
                    speaker_turns_generated = True
                    logger.info("Diarización de turnos de hablante completada.")

                    # --- INICIO: Intento de parche v2 para el URI de Annotation ---
                    if isinstance(diarize_segments_pyannote, Annotation) and \
                       hasattr(diarize_segments_pyannote, 'uri') and \
                       diarize_segments_pyannote.uri == "waveform":
                        try:
                            uri_patch_name = os.path.basename(temp_audio_path)
                            logger.info(f"PARCHE v2: Intentando cambiar URI de Annotation de 'waveform' a '{uri_patch_name}'.")
                            new_annotation = Annotation(uri=uri_patch_name, modality=diarize_segments_pyannote.modality)
                            for segment_pa, track_pa, label_pa in diarize_segments_pyannote.itertracks(yield_label=True): # Renombrar para evitar conflicto
                                new_annotation[segment_pa, track_pa] = label_pa
                            diarize_segments_pyannote = new_annotation
                            logger.debug(f"PARCHE v2: Nuevo URI de diarize_segments_pyannote: {diarize_segments_pyannote.uri}")
                        except Exception as e_rename_uri:
                            logger.warning(f"PARCHE v2: No se pudo cambiar el URI de Annotation: {e_rename_uri}", exc_info=True)
                    # --- FIN: Intento de parche v2 ---

                    if alignment_performed_for_words:
                        current_segments = result.get("segments", [])
                        if isinstance(current_segments, list) and \
                           (len(current_segments) == 0 or (len(current_segments) > 0 and isinstance(current_segments[0], dict) and "words" in current_segments[0])):
                            try:
                                logger.info("Asignando hablantes a palabras...")
                                # logger.debug(f"Type of diarize_segments_pyannote: {type(diarize_segments_pyannote)}, URI: {getattr(diarize_segments_pyannote, 'uri', 'N/A')}")
                                # logger.debug(f"Keys in 'result' for assign_word_speakers: {list(result.keys())}")
                                # logger.debug(f"Sample of first transcript segment for assign: {str(current_segments[0])[:200] if current_segments else 'No segments'}")

                                result_with_speakers = whisperx.assign_word_speakers(diarize_segments_pyannote, result)
                                result = result_with_speakers 
                                word_level_diarization_assigned = True
                                logger.info("Asignación de hablantes a palabras completada.")
                            except KeyError as e_key:
                                tb_str = traceback.format_exc(); msg = f"Fallo en asignación de hablantes (KeyError: {repr(e_key)})."
                                logger.error(msg + f"\nTraceback:\n{tb_str}")
                                response_warnings["diarization_word_assignment"] = msg
                            except Exception as e_assign:
                                tb_str = traceback.format_exc(); msg = f"Error ({type(e_assign).__name__}) en asignación de hablantes: {str(e_assign)}"
                                logger.error(msg + f"\nTraceback:\n{tb_str}", exc_info=False)
                                response_warnings["diarization_word_assignment"] = msg
                        else:
                            msg = "Segmentos de transcripción no tienen 'words' o estructura incorrecta. No se asignan hablantes a palabras."
                            logger.warning(msg + f" (Primer segmento: {str(current_segments[0])[:200] if current_segments else 'Vacío'})")
                            response_warnings["diarization_word_assignment"] = msg
                    else:
                        msg = "Asignación de hablantes a palabras omitida: alineación para palabras no realizada/exitosa."
                        logger.info(msg)
                        if "diarization" not in response_warnings: response_warnings["diarization"] = msg
                except HTTPException: raise
                except Exception as e:
                    msg = f"Error durante la diarización: {str(e)}"; logger.error(msg, exc_info=True)
                    response_warnings["diarization"] = msg
        elif not diarize_audio:
            logger.info("Diarización no solicitada.")

        logger.debug(f"Antes de final_response: type(result)={type(result)}, result_keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}, result_segments_sample={str(result.get('segments', [])[:1])[:200] if isinstance(result, dict) else str(result)[:200]}")

        if not isinstance(result, dict):
            logger.error(f"CRITICAL: 'result' no es un diccionario antes de construir final_response. Tipo: {type(result)}. Valor: {str(result)[:500]}")
            result_segments_for_text = []
            result_text_for_full = ""
            result_lang_for_final = detected_language if detected_language else "Error en detección"
            result_lang_prob_for_final = None
            response_warnings["internal_error_final_result"] = "Estructura de resultado final corrupta."
        else:
            result_segments_for_text = result.get("segments", [])
            if not isinstance(result_segments_for_text, list):
                logger.warning(f"result['segments'] no es una lista, es {type(result_segments_for_text)}. Vaciando para full_text.")
                result_segments_for_text = []
            
            texts = []
            for s_item in result_segments_for_text: # Renombrar s para evitar conflicto con s del nivel superior
                if isinstance(s_item, dict):
                    texts.append(s_item.get('text', '').strip())
                else:
                    logger.warning(f"Elemento en segments no es un dict: {type(s_item)}. Omitiendo para full_text.")
            result_text_for_full = " ".join(texts)
            
            result_lang_for_final = result.get("language", detected_language)
            result_lang_prob_for_final = result.get("language_probability")

        final_response = {
            "language": result_lang_for_final,
            "language_probability": result_lang_prob_for_final,
            "segments": result_segments_for_text,
            "full_text": result.get("text", result_text_for_full) if isinstance(result, dict) else result_text_for_full,
            "options_used": {
                "model_name": model_name, "language_requested": language, "batch_size": batch_size,
                "alignment_requested_for_words": align_audio and word_level_timestamps,
                "alignment_performed_for_words": alignment_performed_for_words,
                "diarization_requested": diarize_audio,
                "diarization_speaker_turns_generated": speaker_turns_generated,
                "diarization_word_level_assigned": word_level_diarization_assigned,
                "word_level_timestamps_requested": word_level_timestamps,
                "device": DEVICE, "compute_type": current_compute_type
            }
        }
        if raw_speaker_activity: final_response["speaker_activity_segments"] = raw_speaker_activity
        if response_warnings: final_response["warnings"] = response_warnings

        return JSONResponse(content=final_response)

    except HTTPException as e_http: # Renombrar para evitar conflicto
        logger.error(f"HTTP Exception: {e_http.detail}")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        if audio_file: audio_file.file.close()
        raise e_http
    except Exception as e_generic: # Renombrar para evitar conflicto
        logger.error(f"Error procesando la solicitud: {e_generic}", exc_info=True)
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        if audio_file: audio_file.file.close()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e_generic)}")
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        if audio_file and not audio_file.file.closed: audio_file.file.close()


@app.get("/health")
async def health_check():
    cuda_ok = True
    if DEVICE == "cuda" and not torch.cuda.is_available():
        cuda_ok = False
        logger.error("Health check: DEVICE is 'cuda' but torch.cuda.is_available() is False!")
    return {
        "status": "ok" if cuda_ok else "degraded",
        "device_in_use": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available_torch": torch.cuda.is_available(),
        "cuda_consistency_ok": cuda_ok,
    }

if __name__ == "__main__":
    import uvicorn
    # Para probar con logs de depuración:
    # logging.getLogger().setLevel(logging.DEBUG) 
    # for handler in logging.getLogger().handlers:
    #    handler.setLevel(logging.DEBUG)
    # logger.info("Nivel de logging establecido en DEBUG para desarrollo local.")
    uvicorn.run(app, host="0.0.0.0", port=8000)