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
from pyannote.core import Annotation 
import traceback

# Configuración de logging
logging.basicConfig(level=logging.INFO) # Cambiar a logging.DEBUG para depuración
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhisperX API (GPU)",
    description="API para transcripción de audio usando WhisperX con alineación y diarización, optimizada para GPU.",
    version="0.1.6" 
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
        except Exception as e: # Capturar cualquier excepción durante la eliminación
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
        if model_key in loaded_models: # Asegurar que no quede una clave parcial si la carga falla
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
        except ImportError: # Específicamente para pyannote.audio no instalado
            logger.error("pyannote.audio no está instalado. Por favor, instálalo para usar diarización.")
            loaded_diarization_pipelines[diarize_key] = None # Marcar como no disponible
            return None
        except Exception as e: # Capturar otras excepciones (token HF, red, etc.)
            logger.error(f"Error cargando pipeline de diarización: {e}", exc_info=True)
            loaded_diarization_pipelines[diarize_key] = None # Marcar como no disponible
            if "401 Client Error" in str(e) or "Unauthorized" in str(e) or "authentication" in str(e).lower():
                 raise HTTPException(status_code=401, detail=f"Error de autenticación al cargar pipeline de diarización: {str(e)}. Verifique su HF_TOKEN.")
            raise HTTPException(status_code=500, detail=f"Error al cargar pipeline de diarización: {str(e)}.")
    
    # Si la pipeline se marcó como None en un intento anterior, devolver None.
    if loaded_diarization_pipelines.get(diarize_key) is None and diarize_key in loaded_diarization_pipelines:
        logger.warning("Pipeline de diarización no pudo ser cargada previamente.")
        return None
        
    return loaded_diarization_pipelines.get(diarize_key)
# --- FIN Funciones auxiliares ---

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
    # Limpieza básica del nombre de archivo para evitar problemas de path
    safe_original_filename = "".join(c for c in original_filename if c.isalnum() or c in ['.', '_', '-']).strip()
    if not safe_original_filename: safe_original_filename = "audio.tmp" # Fallback si el nombre se vacía
    temp_audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{safe_original_filename}")

    response_warnings = {}
    raw_speaker_activity = None
    detected_language = None # Inicializar para asegurar que esté definido

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Archivo de audio guardado en: {temp_audio_path}")

        model = get_model(model_name, DEVICE, current_compute_type, language=language if model_name == "large-v3" else None)

        transcribe_kwargs = {}
        if language: transcribe_kwargs['language'] = language
        
        # model.transcribe de WhisperX devuelve un diccionario
        result: dict = model.transcribe(temp_audio_path, batch_size=batch_size, print_progress=False, **transcribe_kwargs)
        
        # Verificación de la estructura del resultado de la transcripción inicial
        if not isinstance(result, dict) or "language" not in result or "segments" not in result:
            logger.error(f"Resultado inesperado de model.transcribe. Tipo: {type(result)}. Valor: {str(result)[:500]}")
            # No continuar si la estructura básica no está presente
            raise HTTPException(status_code=500, detail="Error interno: la transcripción inicial no devolvió la estructura esperada.")
        
        detected_language = result.get("language") # Usar .get() por si 'language' es None
        logger.info(f"Transcripción inicial completada. Idioma detectado/usado: {detected_language}")

        loaded_audio_data = None
        alignment_performed_for_words = False

        # 3. Alinear transcripción (opcional)
        if align_audio and word_level_timestamps:
            if not detected_language: # Necesitamos un idioma para cargar el modelo de alineación
                logger.warning("No se pudo detectar el idioma, saltando la alineación para marcas de tiempo de palabras.")
                response_warnings["alignment"] = "No se pudo determinar el idioma para la alineación de palabras."
            else:
                try:
                    align_model, align_metadata = get_alignment_model(detected_language, DEVICE)
                    logger.info(f"Alineando transcripción para idioma: {detected_language}...")
                    if loaded_audio_data is None:
                        logger.info("Cargando datos de audio para alineación...")
                        loaded_audio_data = whisperx.load_audio(temp_audio_path)
                    
                    # whisperx.align devuelve una LISTA de segmentos alineados
                    aligned_segments = whisperx.align(
                        result["segments"], # Pasar solo la lista de segmentos
                        align_model,
                        align_metadata,
                        loaded_audio_data,
                        DEVICE,
                        return_char_alignments=False
                    )
                    # Actualizar la clave 'segments' en el diccionario 'result'
                    result["segments"] = aligned_segments
                    
                    # Verificar si la alineación realmente produjo información de palabras
                    if aligned_segments and isinstance(aligned_segments, list) and len(aligned_segments) > 0 and \
                       isinstance(aligned_segments[0], dict) and "words" in aligned_segments[0]:
                        alignment_performed_for_words = True
                        logger.info("Alineación para marcas de tiempo de palabras completada.")
                    else: # La alineación se ejecutó pero no parece haber 'words'
                        logger.warning("Alineación realizada, pero los segmentos no parecen tener información de palabras.")
                        # Loguear más detalles si el nivel es DEBUG
                        if logger.isEnabledFor(logging.DEBUG):
                            if aligned_segments and isinstance(aligned_segments, list) and len(aligned_segments) > 0 and isinstance(aligned_segments[0], dict):
                                logger.debug(f"Primer segmento alineado (sin 'words'?): {str(aligned_segments[0])[:500]}")
                            elif not aligned_segments : # Cubre None o lista vacía
                                logger.debug("aligned_segments está vacío o es None después de whisperx.align")
                            else: # aligned_segments no es lista o el primer elemento no es dict
                                logger.debug(f"aligned_segments tiene estructura inesperada: type={type(aligned_segments)}, len={len(aligned_segments) if isinstance(aligned_segments, list) else 'N/A'}")
                        response_warnings["alignment"] = "Alineación realizada, pero sin información de palabras detectada en los segmentos."
                        
                except HTTPException as e_align_http: # Errores de carga de modelo de alineación
                    logger.warning(f"No se pudo realizar la alineación para marcas de tiempo de palabras: {e_align_http.detail}")
                    response_warnings["alignment"] = f"Alineación de palabras fallida: {e_align_http.detail}"
                except Exception as e_align_generic: # Otros errores durante la alineación
                    logger.error(f"Error inesperado durante la alineación de palabras: {e_align_generic}", exc_info=True)
                    response_warnings["alignment"] = f"Error inesperado en alineación de palabras: {str(e_align_generic)}"
        elif align_audio and not word_level_timestamps:
            logger.info("Alineación solicitada (`align_audio=True`) pero `word_level_timestamps=False`. No se realizará alineación para palabras.")
        elif not align_audio:
            logger.info("Alineación no solicitada (`align_audio=False`).")

        # Verificación defensiva: asegurar que 'result' siga siendo un diccionario después de la alineación
        if not isinstance(result, dict):
            logger.error(f"CRITICAL: 'result' no es un diccionario después de la sección de alineación. Tipo: {type(result)}. Forzando a estructura básica.")
            # Fallback para intentar continuar, aunque los resultados serán limitados
            result = {"segments": [], "language": detected_language if detected_language else None, "text": ""}
            response_warnings["internal_error_structure"] = "Error de estructura interna (post-alineación), resultados pueden ser incompletos."


        # 4. Diarizar (opcional)
        word_level_diarization_assigned = False
        speaker_turns_generated = False

        if diarize_audio:
            diarize_pipeline = None
            try:
                diarize_pipeline = get_diarization_pipeline(hf_token, DEVICE)
            except HTTPException as e_get_diarize: # Errores de carga de pipeline de diarización
                logger.warning(f"No se pudo cargar/obtener la pipeline de diarización: {e_get_diarize.detail}")
                response_warnings["diarization"] = f"Pipeline de diarización no disponible: {e_get_diarize.detail}"
            
            if diarize_pipeline:
                logger.info("Iniciando diarización de turnos de hablante...")
                try:
                    if loaded_audio_data is None: # Cargar audio si no se hizo para alineación
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
                            for segment_pa, track_pa, label_pa in diarize_segments_pyannote.itertracks(yield_label=True):
                                new_annotation[segment_pa, track_pa] = label_pa
                            diarize_segments_pyannote = new_annotation
                            logger.debug(f"PARCHE v2: Nuevo URI de diarize_segments_pyannote: {diarize_segments_pyannote.uri}")
                        except Exception as e_rename_uri:
                            logger.warning(f"PARCHE v2: No se pudo cambiar el URI de Annotation: {e_rename_uri}", exc_info=True)
                    # --- FIN: Intento de parche v2 ---

                    if alignment_performed_for_words:
                        current_segments = result.get("segments", []) # Obtener segmentos del dict 'result'
                        # Verificar que los segmentos tengan la estructura esperada para assign_word_speakers
                        if isinstance(current_segments, list) and \
                           (len(current_segments) == 0 or (len(current_segments) > 0 and isinstance(current_segments[0], dict) and "words" in current_segments[0])):
                            try:
                                logger.info("Asignando hablantes a palabras...")
                                # Logs de depuración (opcionales, controlados por nivel de logging)
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug(f"Type of diarize_segments_pyannote: {type(diarize_segments_pyannote)}, URI: {getattr(diarize_segments_pyannote, 'uri', 'N/A')}")
                                    logger.debug(f"Keys in 'result' for assign_word_speakers: {list(result.keys()) if isinstance(result, dict) else 'result no es dict'}")
                                    logger.debug(f"Sample of first transcript segment for assign: {str(current_segments[0])[:200] if current_segments else 'No segments'}")

                                result_with_speakers = whisperx.assign_word_speakers(diarize_segments_pyannote, result) # 'result' es un dict
                                result = result_with_speakers # assign_word_speakers devuelve un nuevo dict
                                word_level_diarization_assigned = True
                                logger.info("Asignación de hablantes a palabras completada.")
                            except KeyError as e_key:
                                tb_str = traceback.format_exc(); msg = f"Fallo en asignación de hablantes (KeyError: {repr(e_key)})."
                                logger.error(msg + f"\nTraceback:\n{tb_str}") # Loguear el traceback completo
                                response_warnings["diarization_word_assignment"] = msg
                            except Exception as e_assign: # Otras excepciones durante la asignación
                                tb_str = traceback.format_exc(); msg = f"Error ({type(e_assign).__name__}) en asignación de hablantes: {str(e_assign)}"
                                logger.error(msg + f"\nTraceback:\n{tb_str}", exc_info=False) # FastAPI ya loguea con exc_info=True para errores 500
                                response_warnings["diarization_word_assignment"] = msg
                        else: # Los segmentos no tienen la estructura esperada (ej. sin 'words')
                            msg = "Segmentos de transcripción no tienen 'words' o estructura incorrecta. No se asignan hablantes a palabras."
                            logger.warning(msg + f" (Primer segmento: {str(current_segments[0])[:200] if current_segments and isinstance(current_segments,list) and len(current_segments)>0 and isinstance(current_segments[0], dict) else 'Estructura inesperada o vacío'})")
                            response_warnings["diarization_word_assignment"] = msg
                    else: # alignment_performed_for_words es False
                        msg = "Asignación de hablantes a palabras omitida: alineación para palabras no realizada/exitosa."
                        logger.info(msg)
                        if "diarization" not in response_warnings: # No sobrescribir un error más específico
                            response_warnings["diarization"] = msg
                except HTTPException as e_diarize_http: # Re-elevar HTTPExceptions de get_diarization_pipeline
                    raise e_diarize_http
                except Exception as e_diarize_generic: # Otros errores durante la diarización
                    msg = f"Error durante la diarización: {str(e_diarize_generic)}"; logger.error(msg, exc_info=True)
                    response_warnings["diarization"] = msg
        elif not diarize_audio:
            logger.info("Diarización no solicitada.")

        # --- INICIO: Logging de depuración descompuesto para 'result' ---
        if logger.isEnabledFor(logging.DEBUG): # Solo ejecutar si el logging DEBUG está activo
            logger.debug(f"Punto de control A: Iniciando logs pre-final_response.")
            result_type_str = "N/A"
            try: result_type_str = str(type(result))
            except Exception as e_log_type: logger.error(f"Error al obtener type(result) para log: {e_log_type}")
            logger.debug(f"Punto de control B: type(result)={result_type_str}")

            result_keys_str = "N/A (result no es dict)"
            if isinstance(result, dict):
                try: result_keys_str = str(list(result.keys()))
                except TypeError as e_log_keys: 
                    logger.error(f"TypeError al obtener list(result.keys()): {e_log_keys}. Claves podrían contener slices.")
                    result_keys_str = f"Error al listar claves (TypeError): {e_log_keys}"
                except Exception as e_log_keys_other:
                    logger.error(f"Error general al obtener list(result.keys()): {e_log_keys_other}")
                    result_keys_str = f"Error general al listar claves: {e_log_keys_other}"
            logger.debug(f"Punto de control C: result_keys={result_keys_str}")

            result_segments_sample_str = "N/A"
            if isinstance(result, dict):
                segments_for_log = result.get('segments', [])
                if isinstance(segments_for_log, list) and len(segments_for_log) > 0 and isinstance(segments_for_log[0], dict) :
                    try: result_segments_sample_str = str(segments_for_log[0])[:200]
                    except Exception as e_log_seg:
                        logger.error(f"Error al convertir primer segmento a str para log: {e_log_seg}")
                        result_segments_sample_str = f"Error al convertir segmento: {e_log_seg}"
                elif isinstance(segments_for_log, list) and len(segments_for_log) == 0:
                    result_segments_sample_str = "Lista de segmentos vacía"
                else: 
                    result_segments_sample_str = f"result['segments'] estructura inesperada: type={type(segments_for_log)}"
            else: 
                try: result_segments_sample_str = str(result)[:200]
                except Exception as e_log_res_str:
                    logger.error(f"Error al convertir 'result' (no dict) a str para log: {e_log_res_str}")
                    result_segments_sample_str = f"Error al convertir result (no dict): {e_log_res_str}"
            logger.debug(f"Punto de control D: result_segments_sample (o result)={result_segments_sample_str}")
        # --- FIN: Logging de depuración descompuesto ---

        # Construcción de la respuesta final, asegurando que 'result' sea un diccionario
        if not isinstance(result, dict):
            logger.error(f"CRITICAL: 'result' no es un diccionario antes de construir final_response. Tipo: {type(result)}. Valor: {str(result)[:500]}")
            result_segments_for_text = []
            result_text_for_full = ""
            result_lang_for_final = detected_language if detected_language else "Error en detección"
            result_lang_prob_for_final = None
            response_warnings["internal_error_final_result"] = "Estructura de resultado final corrupta."
        else:
            result_segments_for_text = result.get("segments", [])
            if not isinstance(result_segments_for_text, list): # Fallback si 'segments' no es una lista
                logger.warning(f"result['segments'] no es una lista, es {type(result_segments_for_text)}. Vaciando para full_text y segments.")
                result_segments_for_text = []
            
            texts = []
            for s_item in result_segments_for_text: # 's_item' para evitar conflicto con 's' de nivel superior
                if isinstance(s_item, dict):
                    texts.append(s_item.get('text', '').strip())
                else: # Elemento inesperado en la lista de segmentos
                    logger.warning(f"Elemento en segments no es un dict: {type(s_item)}. Omitiendo para full_text.")
            result_text_for_full = " ".join(texts)
            
            result_lang_for_final = result.get("language", detected_language) # Usar lenguaje de 'result' si está, sino el detectado
            result_lang_prob_for_final = result.get("language_probability")


        final_response = {
            "language": result_lang_for_final,
            "language_probability": result_lang_prob_for_final,
            "segments": result_segments_for_text, # Ya es una lista (o lista vacía)
            "full_text": result.get("text", result_text_for_full) if isinstance(result, dict) else result_text_for_full, # Usar result.get("text") si existe, sino el reconstruido
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

    except HTTPException as e_http: # Renombrar para evitar conflicto de variable 'e'
        logger.error(f"HTTP Exception: {e_http.detail}")
        # Limpieza en caso de HTTPException
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        if audio_file: audio_file.file.close()
        raise e_http # Re-elevar la excepción
    except Exception as e_generic: # Renombrar para evitar conflicto de variable 'e'
        logger.error(f"Error procesando la solicitud: {e_generic}", exc_info=True)
        # Limpieza en caso de Exception genérica
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        if audio_file: audio_file.file.close()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e_generic)}")
    finally:
        # Bloque finally para asegurar limpieza, incluso si ya se hizo en un except
        if 'temp_dir' in locals() and os.path.exists(temp_dir): # locals() para asegurar que temp_dir fue definido
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal {temp_dir} eliminado (bloque finally).")
        if audio_file and hasattr(audio_file, 'file') and not audio_file.file.closed: # Más robusto
            audio_file.file.close()

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
    # Para desarrollo local, descomentar para logs DEBUG:
    # logging.getLogger().setLevel(logging.DEBUG)
    # for handler in logging.getLogger().handlers:
    #    handler.setLevel(logging.DEBUG)
    # logger.info("Nivel de logging establecido en DEBUG para desarrollo local.")
    uvicorn.run(app, host="0.0.0.0", port=8000)