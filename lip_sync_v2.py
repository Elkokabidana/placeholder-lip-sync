# --- START OF FILE lip_sync_v11_gan_alphablend_final.py --- # نام پیشنهادی
import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import queue
import librosa
import sys
import time
import os
import torch
import torchvision.transforms as transforms
from collections import deque
import traceback

# --- تنظیمات اولیه ---
WAV2LIP_DIR = 'Wav2Lip'
WAV2LIP_ABS_PATH = os.path.abspath(WAV2LIP_DIR)
if WAV2LIP_ABS_PATH not in sys.path: sys.path.insert(0, WAV2LIP_ABS_PATH); print(f"Path '{WAV2LIP_ABS_PATH}' added.")
try: from models import Wav2Lip; print("Wav2Lip models imported successfully.")
except ImportError as e: print(f"Import Error: {e}"); sys.exit(1)
except Exception as e: print(f"Unexpected Import Error: {e}"); sys.exit(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# *** استفاده از مدل GAN ***
WAV2LIP_MODEL_PATH = os.path.join(WAV2LIP_DIR, 'checkpoints', 'wav2lip_gan.pth')
print(f"Model path set to (GAN MODEL): {WAV2LIP_MODEL_PATH}")
if not os.path.exists(WAV2LIP_MODEL_PATH): print(f"Error: GAN Model file not found at '{WAV2LIP_MODEL_PATH}'!"); sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f"Using device: {DEVICE}")
if DEVICE == 'cpu': print("Warning: CPU execution will be very slow.")

# --- پارامترهای اصلی ---
IMG_SIZE = 96; MEL_STEP_SIZE = 16; FPS = 25; WAV2LIP_BATCH_SIZE = 2
WEBCAM_WIDTH = 640; WEBCAM_HEIGHT = 480; FORMAT = pyaudio.paInt16
CHANNELS = 1; RATE = 16000; CHUNK = int(RATE / FPS); INPUT_DEVICE_INDEX = None

# --- صف و رویداد ---
audio_queue = queue.Queue(maxsize=int(FPS*1.5)); exit_event = threading.Event()

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- بافرها ---
hop_length = 200
AUDIO_BUFFER_MAXLEN = int(FPS * 3); print(f"Audio buffer max length set to: {AUDIO_BUFFER_MAXLEN} chunks")
audio_buffer = deque(maxlen=AUDIO_BUFFER_MAXLEN)
frame_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE); bbox_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE)
original_frame_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE)

# --- توابع کمکی ---
# (تابع face_detect_with_landmarks و بقیه مثل قبل)
def face_detect_with_landmarks(images): # (بدون تغییر)
    all_landmarks = []; image_shapes = []
    for image in images:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
            results = face_mesh.process(image_rgb); image_rgb.flags.writeable = True
            all_landmarks.append(results.multi_face_landmarks); image_shapes.append(image.shape)
        except Exception as e: all_landmarks.append(None); image_shapes.append(None)
    bboxes = []
    for i, face_landmarks_list in enumerate(all_landmarks):
        shape = image_shapes[i]
        if not face_landmarks_list or shape is None: bboxes.append(None); continue
        face_landmarks = face_landmarks_list[0]; ih, iw, _ = shape
        try:
            lm_mouth_left = face_landmarks.landmark[61]; lm_mouth_right = face_landmarks.landmark[291]
            lm_lip_upper = face_landmarks.landmark[0]; lm_lip_lower = face_landmarks.landmark[17]
            mouth_left_x = int(lm_mouth_left.x * iw); mouth_right_x = int(lm_mouth_right.x * iw)
            lip_upper_y = int(lm_lip_upper.y * ih); lip_lower_y = int(lm_lip_lower.y * ih)
            mouth_cx = (mouth_left_x + mouth_right_x) // 2; mouth_cy = (lip_upper_y + lip_lower_y) // 2
            mouth_width = mouth_right_x - mouth_left_x
            if mouth_width <= 0: bboxes.append(None); continue
            size = int(mouth_width * 2.5); center_x = mouth_cx; center_y = mouth_cy + int(size * 0.1)
        except Exception as e_lm: bboxes.append(None); continue
        half_size = size // 2; x1 = max(0, center_x - half_size); y1 = max(0, center_y - half_size)
        x2 = min(iw, center_x + half_size); y2 = min(ih, center_y + half_size)
        if (x2 - x1) > 0 and (y2 - y1) > 0: bboxes.append([x1, y1, x2, y2])
        else: bboxes.append(None)
    return bboxes

def preprocess_frames(frames, bboxes): # (بدون تغییر)
    preprocessed_frames = []; valid_indices = []
    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        if bbox is None: continue
        x1, y1, x2, y2 = map(int, bbox)
        if x1 >= x2 or y1 >= y2: continue
        face_crop = frame[y1:y2, x1:x2];
        if face_crop.size == 0: continue
        try:
             face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA); face_resized_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
             face_tensor = torch.FloatTensor(face_resized_rgb).permute(2, 0, 1) / 255.0; face_normalized = (face_tensor - 0.5) * 2.0
             preprocessed_frames.append(face_normalized); valid_indices.append(i)
        except Exception as e: continue
    if not preprocessed_frames: return None, None
    try: return torch.stack(preprocessed_frames).unsqueeze(0), valid_indices
    except Exception as e: print(f"Stacking Error: {e}"); return None, None

def get_mel_chunk(audio_data_bytes): # (بدون تغییر)
    try: audio_signal = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0;
    except ValueError as e: print(f"Audio Conv Error: {e}"); return None
    if len(audio_signal) < hop_length: return None
    n_fft, win_length, n_mels = 800, 800, 80; fmin, fmax = 55, 7600
    expected_signal_len = MEL_STEP_SIZE * hop_length
    if len(audio_signal) < expected_signal_len: audio_signal = np.pad(audio_signal, (0, expected_signal_len - len(audio_signal)), 'constant')
    elif len(audio_signal) > expected_signal_len: audio_signal = audio_signal[:expected_signal_len]
    try:
        mel = librosa.feature.melspectrogram(y=audio_signal, sr=RATE, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax, center=False)
        mel_db = librosa.power_to_db(mel, ref=np.max)
    except Exception as e: print(f"Librosa Error: {e}"); return None
    if mel_db.shape[1] != MEL_STEP_SIZE:
        target_len = MEL_STEP_SIZE; current_len = mel_db.shape[1]
        if current_len < target_len: mel_db = np.pad(mel_db, ((0, 0), (0, target_len - current_len)), mode='constant', constant_values=-80.0)
        elif current_len > target_len: mel_db = mel_db[:, :target_len]
    if not np.isfinite(mel_db).all(): mel_db = np.nan_to_num(mel_db, nan=-80.0, posinf=-80.0, neginf=-80.0)
    return torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)

def list_audio_input_devices(): # (بدون تغییر)
    p = pyaudio.PyAudio(); numdevices=0; available_indices = []
    try: info = p.get_host_api_info_by_index(0); numdevices = info.get('deviceCount', 0)
    except Exception as e: print(f"Host API Error: {e}"); p.terminate(); return []
    print("-" * 60 + "\nAvailable Audio Input Devices:\n" + "-" * 60)
    found_input_device = False
    for i in range(0, numdevices):
        try:
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                found_input_device = True; available_indices.append(i); device_name_raw = device_info.get('name'); device_name = "Unknown"
                if isinstance(device_name_raw, bytes):
                    try: device_name = device_name_raw.decode('utf-8', errors='replace')
                    except UnicodeDecodeError: device_name = device_name_raw.decode('latin-1', errors='replace')
                elif isinstance(device_name_raw, str): device_name = device_name_raw
                print(f"  Index {i}: {device_name}")
        except Exception as e: print(f"  Dev Info Err {i}: {e}")
    if not found_input_device: print("No active input devices found.")
    print("-" * 60); p.terminate(); return available_indices

def record_audio(device_index, stop_event): # (بدون تغییر)
    p = pyaudio.PyAudio(); stream = None
    try: stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=device_index); print(f"🎤 Recording started from Index {device_index}...")
    except Exception as e: print(f"Stream Open Err: {e}"); p.terminate(); stop_event.set(); return
    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False);
            if stop_event.is_set(): break
            try: audio_queue.put(data, block=True, timeout=0.1)
            except queue.Full:
                 try: audio_queue.get_nowait(); audio_queue.put_nowait(data)
                 except Exception: pass
        except IOError as e:
            if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed: time.sleep(0.001)
            elif hasattr(e, 'errno') and e.errno == pyaudio.paStreamIsStopped: print("Audio stream stopped."); stop_event.set(); break
            else: print(f"IO Err Aud: {e}"); time.sleep(0.01)
        except Exception as e: print(f"Unknown Err Aud: {e}"); stop_event.set(); break
    print(f"🎤 Recording stopped for Index {device_index}.")
    if stream:
        try:
            if stream.is_active(): stream.stop_stream()
            stream.close()
        except Exception as e: print(f"Error closing audio stream: {e}")
    try: p.terminate()
    except Exception as e: print(f"Error terminating PyAudio: {e}")
    print("Audio recording thread finished.")

# --- تابع اصلی پردازش ویدیو (با مدل GAN، تراز بندی دهان، Alpha Blend، اصلاح رنگ) ---
def process_video(stop_event, model):
    global audio_buffer, frame_buffer, bbox_buffer, original_frame_buffer
    cap = cv2.VideoCapture(0);
    if not cap.isOpened(): print("Error: Cannot open webcam."); stop_event.set(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT); cap.set(cv2.CAP_PROP_FPS, FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS); effective_fps = actual_fps if actual_fps > 0 else FPS; print(f"Requested FPS: {FPS}, Actual FPS: {effective_fps}")
    # --- *** پارامترهای قابل تنظیم برای ترکیب *** ---
    MASK_START_RATIO = 0.65  # <--- این مقدار را افزایش بده (مثلا 0.6, 0.65)
    MASK_BLUR_KERNEL_SIZE = (31, 31)  # <--- این مقدار را افزایش بده (مثلا 21, 31, 41 - باید فرد باشد)
    # --- *** ---

    # پیش محاسبه ماسک برای Alpha Blending
    try:
        base_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32); mask_start_row = int(IMG_SIZE * MASK_START_RATIO); base_mask[mask_start_row:, :] = 1.0; mask_blurred_precalculated = base_mask
        if (MASK_BLUR_KERNEL_SIZE[0] > 0 and MASK_BLUR_KERNEL_SIZE[1] > 0 and MASK_BLUR_KERNEL_SIZE[0]%2!=0 and MASK_BLUR_KERNEL_SIZE[1]%2!=0):
            mask_blurred_precalculated = cv2.GaussianBlur(base_mask, MASK_BLUR_KERNEL_SIZE, 0); print("Pre-calculated blurred mask.")
        elif MASK_BLUR_KERNEL_SIZE != (0, 0): print("Warning: Invalid mask blur kernel size. Blur disabled.")
    except Exception as e: print(f"Mask Precomputation Error: {e}"); mask_blurred_precalculated = np.ones((IMG_SIZE,IMG_SIZE), dtype=np.float32)*0.5 # Fallback

    print("📸 Video processing and Lip Sync started...")
    display_counter = 0; processing_times = deque(maxlen=int(effective_fps * 2)); last_display_time = time.time(); last_known_good_bbox = None; generated_face_cache = {}

    while cap.isOpened() and not stop_event.is_set():
        # (دریافت صدا، فریم، تشخیص چهره با لندمارک)
        audio_read_count = 0;
        while not audio_queue.empty() and audio_read_count < WAV2LIP_BATCH_SIZE * 2 :
            try: audio_buffer.append(audio_queue.get_nowait()); audio_read_count += 1
            except queue.Empty: break
        ret, frame = cap.read();
        if not ret: time.sleep(0.05); continue
        frame = cv2.flip(frame, 1)
        current_bbox = face_detect_with_landmarks([frame])[0];
        if current_bbox is not None: last_known_good_bbox = current_bbox
        else: current_bbox = last_known_good_bbox
        if current_bbox is None:
            cv2.imshow('Real-time Lip Sync (GAN - AlphaBlend Final)', frame) # نام پنجره
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): print("'q' pressed while no face detected. Exiting..."); stop_event.set(); break
            continue

        # (اضافه کردن به بافرها)
        frame_buffer.append(frame.copy()); original_frame_buffer.append(frame.copy()); bbox_buffer.append(current_bbox)

        output_frame = None
        if len(frame_buffer) == WAV2LIP_BATCH_SIZE: # Level 2
            frames_to_process = list(frame_buffer); bboxes_to_process = list(bbox_buffer)
            face_batch, valid_indices = preprocess_frames(frames_to_process, bboxes_to_process)

            # (پردازش صدا)
            bytes_per_sample = 2; num_samples_needed = MEL_STEP_SIZE * hop_length; num_bytes_needed = num_samples_needed * bytes_per_sample * CHANNELS
            bytes_per_chunk = CHUNK * bytes_per_sample * CHANNELS; num_chunks_to_take = int(np.ceil(num_bytes_needed / bytes_per_chunk)) if bytes_per_chunk > 0 else 1
            num_chunks_in_buffer = len(audio_buffer)
            mel_chunk = None
            if num_chunks_in_buffer >= num_chunks_to_take:
                audio_segment_chunks = list(audio_buffer)[-num_chunks_to_take:]; audio_segment_bytes = b''.join(audio_segment_chunks)
                if len(audio_segment_bytes) >= num_bytes_needed:
                    input_bytes_for_mel = audio_segment_bytes[-num_bytes_needed:]
                    mel_chunk = get_mel_chunk(input_bytes_for_mel)
                    if mel_chunk is not None: mel_chunk = mel_chunk.to(DEVICE)

            generated_faces = None; generated_face_cache.clear()
            if face_batch is not None and mel_chunk is not None and valid_indices: # Level 3
                face_batch = face_batch.to(DEVICE);
                with torch.no_grad(): # Level 4
                    try: # Level 5
                        generated_faces_batch = model(mel_chunk, face_batch);
                        generated_faces = generated_faces_batch.squeeze(0).cpu().numpy();
                        generated_faces = np.transpose(generated_faces, (0, 2, 3, 1))
                        # *** دنرمالیزاسیون برای مدل GAN ***
                        generated_faces = np.clip((generated_faces + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                        # *** پایان دنرمالیزاسیون ***
                        for i, face_idx in enumerate(valid_indices):
                            if i < len(generated_faces): generated_face_cache[face_idx] = generated_faces[i]
                    except Exception as e: # Level 5
                        print(f"!!!!!!!!! Model Execution Error: {e} !!!!!!!!!"); traceback.print_exc()

            # --- ترکیب Alpha Blending (با مدل GAN و اصلاح رنگ) ---
            output_frame = original_frame_buffer[0].copy(); # Level 3
            output_bbox = bboxes_to_process[0]; output_face_to_paste = generated_face_cache.get(0)

            # (نمایش خروجی خام - با cvtColor)
            # می توان این بخش را برای اجرای نهایی کامنت کرد
            if output_face_to_paste is not None: # Level 3
                try: # Level 4
                    # فرض: خروجی مدل GAN RGB است
                    raw_face_display = cv2.cvtColor(output_face_to_paste, cv2.COLOR_RGB2BGR) # <-- بازگردانده شد
                    cv2.imshow("Raw Model Output (96x96) - GAN Model", raw_face_display)
                except Exception as e_raw: print(f"Error showing raw model output: {e_raw}") # Level 4

            if output_face_to_paste is not None and output_bbox is not None: # Level 3
                x1, y1, x2, y2 = map(int, output_bbox);
                x1, y1 = max(x1, 0), max(y1, 0);
                x2, y2 = min(x2, output_frame.shape[1]), min(y2, output_frame.shape[0])
                if x1 < x2 and y1 < y2: # Level 4
                    target_h, target_w = y2 - y1, x2 - x1
                    try: # Level 4 - Alpha Blending Block
                        # 1. تغییر اندازه چهره تولیدی
                        gen_face_resized = cv2.resize(output_face_to_paste, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4) # Level 5

                        # 2. *** تبدیل رنگ به BGR ***
                        try: gen_face_resized_bgr = cv2.cvtColor(gen_face_resized, cv2.COLOR_RGB2BGR) # <-- بازگردانده شد
                        except cv2.error: print("Warning: Color conversion failed."); gen_face_resized_bgr = gen_face_resized

                        # 3. تغییر اندازه ماسک محو شده
                        mask_resized = cv2.resize(mask_blurred_precalculated, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        mask_resized = mask_resized[:, :, np.newaxis]

                        # 4. گرفتن ناحیه اصلی
                        original_face_roi = output_frame[y1:y2, x1:x2]

                        # 5. انجام Alpha Blending
                        blended_face = np.clip(
                            original_face_roi.astype(np.float32) * (1.0 - mask_resized) +
                            gen_face_resized_bgr.astype(np.float32) * mask_resized,
                            0, 255
                        )

                        # 6. قرار دادن نتیجه
                        output_frame[y1:y2, x1:x2] = blended_face.astype(np.uint8)

                    except Exception as e: print(f"Blending Error: {e}") # Level 4

            elif output_bbox is not None: # Level 3: Sync? text
                 x1_text, y1_text, _, _ = map(int, output_bbox) # Level 4
                 cv2.putText(output_frame, "Sync?", (max(0,x1_text), max(0, y1_text-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1) # Level 4

            # --- پاک کردن بافرها --- (Level 3)
            frame_buffer.popleft(); bbox_buffer.popleft(); original_frame_buffer.popleft()
        # --- پایان بلوک اصلی پردازش بچ ---

        else:  # Buffering... (Level 2)
            output_frame = frame.copy()
            cv2.putText(output_frame, f"Buffering... {len(frame_buffer)}/{WAV2LIP_BATCH_SIZE}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # --- نمایش --- (Level 2)
        if output_frame is not None:
            cv2.imshow('Real-time Lip Sync (GAN - AlphaBlend Final)', output_frame) # نام پنجره
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): print("'q' pressed. Exiting..."); stop_event.set(); break

    # --- تمیزکاری --- (Level 1)
    print("Releasing webcam and destroying windows..."); cap.release(); cv2.destroyAllWindows();
    if 'face_mesh' in locals() and hasattr(face_mesh, 'close'):
        try: face_mesh.close()
        except Exception as e: print(f"Error closing face mesh: {e}")
    print("Video processing thread finished.")


# --- اجرای اصلی برنامه ---
# (بدون تغییر)
if __name__ == "__main__":
    print("="*30 + " Audio Device Selection " + "="*30); available_indices = list_audio_input_devices(); INPUT_DEVICE_INDEX = None
    if not available_indices: print("Error: No input devices found."); sys.exit(1)
    if len(available_indices) == 1: INPUT_DEVICE_INDEX = available_indices[0]; print(f"Auto-selected Index: {INPUT_DEVICE_INDEX}")
    else:
        while INPUT_DEVICE_INDEX is None:
             try: selected = input(f"Enter microphone Index {available_indices}: "); candidate_index = int(selected.strip())
             except ValueError: print("Invalid input."); continue
             except (EOFError, KeyboardInterrupt): print("\nSelection cancelled."); sys.exit(0)
             if candidate_index in available_indices: INPUT_DEVICE_INDEX = candidate_index
             else: print("Invalid Index.")
    print(f"--> Selected Audio Device Index: {INPUT_DEVICE_INDEX}")
    print("="*70)
    print("="*30 + " Model Loading " + "="*30); print(f"Loading GAN model from: {WAV2LIP_MODEL_PATH}...")
    model = None
    try:
        model = Wav2Lip(); checkpoint = torch.load(WAV2LIP_MODEL_PATH, map_location=DEVICE)
        if "state_dict" in checkpoint: s = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict): s = checkpoint
        else: print("Error: Unknown checkpoint format."); sys.exit(1)
        new_s = {};
        for k, v in s.items(): new_s[k.replace('module.', '', 1)] = v
        model.load_state_dict(new_s); print("Model loaded successfully.")
    except Exception as e: print(f"Model Loading Error: {e}"); traceback.print_exc(); sys.exit(1)
    model = model.to(DEVICE); model.eval(); print(f"Model moved to '{DEVICE}' and set to eval mode.")
    print("="*70)
    print("="*30 + " Starting Threads " + "="*30); print("Starting audio recording thread...");
    audio_thread = threading.Thread(target=record_audio, args=(INPUT_DEVICE_INDEX, exit_event), daemon=True)
    audio_thread.start(); time.sleep(2)
    if not audio_thread.is_alive(): print("Error: Audio thread failed to start."); sys.exit(1)
    print("Audio thread started.")
    print("="*70)
    print("="*30 + " Starting Processing " + "="*30); video_processing_successful = True
    try:
        print("Performing Model Warm-up...");
        try:
            dummy_mel = torch.randn(1, 1, 80, MEL_STEP_SIZE, device=DEVICE)
            dummy_face_batch = torch.randn(1, WAV2LIP_BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
            with torch.no_grad(): _ = model(dummy_mel, dummy_face_batch)
            print("Warm-up complete.")
        except Exception as warmup_e: print(f"Warning: Warm-up failed: {warmup_e}. Continuing...")
        print("="*70); process_video(exit_event, model)
    except KeyboardInterrupt: print("\nUser interrupt."); video_processing_successful = False
    except Exception as e: print(f"\nCritical Error: {e}"); traceback.print_exc(); video_processing_successful = False
    finally:
        print("\n" + "="*30 + " Cleaning Up and Exiting " + "="*30); exit_event.set()
        if 'audio_thread' in locals() and audio_thread.is_alive(): print("Waiting for audio thread..."); audio_thread.join(timeout=2.0);
        if 'audio_thread' in locals() and audio_thread.is_alive(): print("Warning: Audio thread did not terminate.")
        status_message = "OK" if video_processing_successful else "Error/Interrupted"
        print(f"Program finished. Final Status: {status_message}")
        print("="*70)

# --- END OF FILE lip_sync_v11_gan_alphablend_final.py ---