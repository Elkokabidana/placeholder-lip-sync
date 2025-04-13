# --- START OF FILE lip_sync_v2_fixed.py ---
# ... (import ها و تنظیمات اولیه مثل قبل) ...
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

WAV2LIP_DIR = 'Wav2Lip'
WAV2LIP_ABS_PATH = os.path.abspath(WAV2LIP_DIR)
if WAV2LIP_ABS_PATH not in sys.path:
    sys.path.insert(0, WAV2LIP_ABS_PATH)
    print(f"مسیر '{WAV2LIP_ABS_PATH}' به sys.path اضافه شد.")

try:
    from models import Wav2Lip
    print("فایل models.py مربوط به Wav2Lip با موفقیت وارد شد.")
except ImportError as e:
    print(f"خطا در وارد کردن Wav2Lip: {e}")
    sys.exit(1)
except Exception as e:
    print(f"خطای غیرمنتظره هنگام وارد کردن models.py: {e}")
    sys.exit(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
WAV2LIP_MODEL_PATH = os.path.join(WAV2LIP_DIR, 'checkpoints', 'wav2lip.pth')
print(f"مسیر مدل Wav2Lip تنظیم شد به: {WAV2LIP_MODEL_PATH}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"دستگاه پردازش: {DEVICE}")
if DEVICE == 'cuda' and not torch.cuda.is_available(): DEVICE = 'cpu'; print("CUDA یافت نشد، استفاده از CPU.")
elif DEVICE == 'cpu': print("هشدار: اجرای روی CPU کند خواهد بود.")

IMG_SIZE = 96
MEL_STEP_SIZE = 16
FPS = 25
WAV2LIP_BATCH_SIZE = 5
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / FPS)
INPUT_DEVICE_INDEX = None
audio_queue = queue.Queue(maxsize=25)
exit_event = threading.Event()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

AUDIO_BUFFER_SIZE_SECONDS = 0.8
AUDIO_BUFFER_MAXLEN = int(AUDIO_BUFFER_SIZE_SECONDS * FPS)
audio_buffer = deque(maxlen=AUDIO_BUFFER_MAXLEN)
frame_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE)
bbox_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE)
original_frame_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE)

# --- توابع کمکی (face_detect, preprocess_frames, get_mel_chunk مثل قبل) ---
def face_detect(images):
    # ... (کد بدون تغییر) ...
    results = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        res = face_detection.process(image_rgb)
        image_rgb.flags.writeable = True
        detections = []
        if res.detections:
            for detection in res.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                xmin = bboxC.xmin if bboxC.xmin is not None else 0
                ymin = bboxC.ymin if bboxC.ymin is not None else 0
                width = bboxC.width if bboxC.width is not None else 0
                height = bboxC.height if bboxC.height is not None else 0
                x, y, w, h = int(xmin * iw), int(ymin * ih), int(width * iw), int(height * ih)
                padding_factor = 0.15
                center_x, center_y = x + w // 2, y + h // 2
                size = max(w, h)
                half_size = size // 2
                pad = int(size * padding_factor)
                half_size += pad
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(iw, center_x + half_size)
                y2 = min(ih, center_y + half_size)
                detections.append([x1, y1, x2, y2])
        results.append(detections)

    bboxes = []
    for dets in results:
        if not dets: bboxes.append(None)
        else:
            try: best_det = max(dets, key=lambda b: (b[2]-b[0])*(b[3]-b[1]) if (b and len(b)==4) else 0); bboxes.append(best_det)
            except Exception: bboxes.append(None)
    return bboxes

def preprocess_frames(frames, bboxes):
    # ... (کد بدون تغییر) ...
    preprocessed_frames = []
    valid_indices = []
    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        if bbox is None: continue
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x1 >= x2 or y1 >= y2: continue
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0: continue
        try:
             face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
             face_resized_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
             face_tensor = torch.FloatTensor(face_resized_rgb).permute(2, 0, 1) / 255.0
             face_normalized = (face_tensor - 0.5) * 2.0
             preprocessed_frames.append(face_normalized)
             valid_indices.append(i)
        except cv2.error as e: print(f"خطای OpenCV در preprocess_frames: {e}"); continue
        except Exception as e: print(f"خطای دیگر در preprocess_frames: {e}"); continue

    if not preprocessed_frames: return None, None
    try:
        batch_tensor = torch.stack(preprocessed_frames).unsqueeze(0) # Shape: (1, B, C, H, W) - **5D**
        # print(f"DEBUG: Preprocessed face batch shape: {batch_tensor.shape}") # Debug shape
        return batch_tensor, valid_indices
    except RuntimeError as e: print(f"خطای RuntimeError در stack: {e}"); return None, None

def get_mel_chunk(audio_data_bytes):
    """تبدیل بایت‌های صدا به مل‌اسپکتروگرام مورد نیاز Wav2Lip (با power_to_db)"""
    try:
        audio_signal = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    except ValueError as e:
        print(f"خطا در تبدیل بایت صدا: {e}, طول بایت‌ها: {len(audio_data_bytes)}")
        return None # در صورت خطای تبدیل

    # پارامترهای استاندارد Wav2Lip
    n_fft, hop_length, win_length, n_mels = 800, 200, 800, 80
    fmin, fmax = 55, 7600
    # طول دقیق سیگنال مورد نیاز برای MEL_STEP_SIZE = 16 فریم با hop_length = 200
    # (mel_steps - 1) * hop + win <= len
    # برای اطمینان، کمی بیشتر می‌گیریم (مثلاً 0.2 ثانیه کامل)
    expected_signal_len = int(0.2 * RATE) # 3200 samples for 0.2 sec

    # پدینگ یا برش سیگنال صوتی
    if len(audio_signal) < expected_signal_len:
        # print(f"پدینگ صدا: {len(audio_signal)} -> {expected_signal_len}")
        audio_signal = np.pad(audio_signal, (0, expected_signal_len - len(audio_signal)), 'constant', constant_values=0.0)
    elif len(audio_signal) > expected_signal_len:
        # print(f"برش صدا: {len(audio_signal)} -> {expected_signal_len}")
        audio_signal = audio_signal[-expected_signal_len:]

    try:
        # محاسبه Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio_signal, sr=RATE, n_fft=n_fft,
                                             hop_length=hop_length, win_length=win_length,
                                             n_mels=n_mels, fmin=fmin, fmax=fmax)
        # تبدیل به دسی‌بل
        # استفاده از ref=np.max رایج است تا مقادیر نسبت به بلندترین بخش نرمال شوند
        mel_db = librosa.power_to_db(mel, ref=np.max)

    except Exception as e:
        print(f"خطا در محاسبه Mel Spectrogram: {e}")
        return None # بازگشت None در صورت خطا

    # پدینگ یا برش Mel Spectrogram به طول MEL_STEP_SIZE
    db_pad_value = -80.0 # مقدار دسی‌بل بسیار پایین برای پدینگ
    if mel_db.shape[1] < MEL_STEP_SIZE:
        shortage = MEL_STEP_SIZE - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, shortage)), mode='constant', constant_values=db_pad_value)
        # print(f"پدینگ Mel: {mel_db.shape[1] - shortage} -> {MEL_STEP_SIZE}")
    elif mel_db.shape[1] > MEL_STEP_SIZE:
         # print(f"برش Mel: {mel_db.shape[1]} -> {MEL_STEP_SIZE}")
         mel_db = mel_db[:, :MEL_STEP_SIZE]


    # اطمینان از وجود مقادیر محدود (نه NaN یا Inf)
    if not np.isfinite(mel_db).all():
        print("هشدار: مقادیر نامحدود در mel_db یافت شد! جایگزینی با 0.")
        mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0)


    # تبدیل به تنسور PyTorch
    mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, n_mels, mel_step_size)
    return mel_tensor


# --- اصلاح تابع list_audio_input_devices (رفع خطای decode) ---
def list_audio_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount', 0)
    print("-" * 60); print("دستگاه‌های ورودی صوتی یافت شده:"); print("-" * 60)
    found_input_device = False
    available_indices = []
    for i in range(0, numdevices):
        try:
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                found_input_device = True
                available_indices.append(i)
                device_name_raw = device_info.get('name') # دریافت نام خام
                device_name = "Unknown" # مقدار پیشفرض
                if isinstance(device_name_raw, bytes): # اگر بایت بود، decode کن
                    try: device_name = device_name_raw.decode('utf-8', errors='replace')
                    except UnicodeDecodeError: device_name = device_name_raw.decode('latin-1', errors='replace')
                elif isinstance(device_name_raw, str): # اگر از قبل رشته بود، مستقیم استفاده کن
                    device_name = device_name_raw
                print(f"  Index {i}: {device_name}")
        except Exception as e:
            # چاپ خطای دقیق تر
            import traceback
            print(f"  خطا در پردازش اطلاعات دستگاه Index {i}: {e}")
            # traceback.print_exc() # برای دیباگ بیشتر می توان فعال کرد
    if not found_input_device: print("هیچ دستگاه ورودی صوتی فعالی یافت نشد.")
    print("-" * 60)
    try: p.terminate()
    except Exception as e: print(f"خطا هنگام بستن PyAudio: {e}")
    return available_indices
# --- پایان اصلاح تابع ---

# --- تابع ضبط صوت (بدون تغییر) ---
def record_audio(device_index, stop_event):
    # ... (کد مثل قبل) ...
    p = pyaudio.PyAudio(); stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=device_index)
        print(f"🎤 ضبط صوت از Index {device_index} با Rate={RATE}, Chunk={CHUNK} آغاز شد...")
    except IOError as e: print(f"خطا باز کردن stream: {e}"); p.terminate(); stop_event.set(); return

    overflow_count = 0; max_overflow_report = 5
    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if stop_event.is_set(): break
            try: audio_queue.put(data, block=False)
            except queue.Full:
                try: audio_queue.get_nowait(); audio_queue.put_nowait(data)
                except queue.Empty: pass
                except queue.Full: pass
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                overflow_count += 1
                if overflow_count <= max_overflow_report: print(f"Warning: Input overflowed ({overflow_count}).")
                time.sleep(0.001)
            elif e.errno == pyaudio.paStreamIsStopped: print("Audio stream stopped."); stop_event.set(); break
            else: print(f"خطای IO صدا: {e} (errno: {getattr(e, 'errno', 'N/A')})"); time.sleep(0.01)
        except Exception as e: print(f"خطای ناشناخته صدا: {e}"); stop_event.set(); break
    print(f"🎤 ضبط صوت از Index {device_index} متوقف شد.")
    if stream:
        try:
             if stream.is_active(): stream.stop_stream()
             stream.close()
        except Exception as e: print(f"خطا بستن stream: {e}")
    try: p.terminate()
    except Exception as e: print(f"خطا بستن PyAudio: {e}")
    print("ترد ضبط صدا خاتمه یافت.")


# --- تابع پردازش ویدیو (اصلاح شده برای خطای ابعاد و UnboundLocalError) ---
def process_video(stop_event, model):
    global audio_buffer, frame_buffer, bbox_buffer, original_frame_buffer

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("خطا باز کردن وب کم."); stop_event.set(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS); print(f"FPS وبکم: {actual_fps if actual_fps > 0 else FPS}")
    if actual_fps <= 0 : actual_fps = FPS

    print("📸 پردازش تصویر و Lip Sync آغاز شد...")
    display_counter = 0; processing_times = deque(maxlen=int(actual_fps))
    last_display_time = time.time(); last_known_good_bbox = None
    generated_face_cache = {}

    while cap.isOpened() and not stop_event.is_set():
        loop_start_time = time.time()

        # --- دریافت صدا و فریم ---
        audio_read_count = 0
        while not audio_queue.empty() and audio_read_count < 5:
            try: audio_buffer.append(audio_queue.get_nowait()); audio_read_count += 1
            except queue.Empty: break
        ret, frame = cap.read()
        if not ret: print("خطا خواندن فریم."); time.sleep(0.05); continue
        frame = cv2.flip(frame, 1)

        # --- تشخیص چهره ---
        current_bbox = face_detect([frame])[0]
        if current_bbox is not None: last_known_good_bbox = current_bbox
        else: current_bbox = last_known_good_bbox

        # --- اضافه کردن به بافرها ---
        frame_buffer.append(frame.copy())
        original_frame_buffer.append(frame.copy())
        bbox_buffer.append(current_bbox)

        # --- اجرای مدل ---
        output_frame = None
        if len(frame_buffer) == WAV2LIP_BATCH_SIZE:
            process_start_time = time.time()

            # --- آماده‌سازی ورودی‌ها ---
            frames_to_process = list(frame_buffer)
            bboxes_to_process = list(bbox_buffer)
            face_batch, valid_indices = preprocess_frames(frames_to_process, bboxes_to_process) # انتظار 5D

            num_chunks_needed = int((0.2 * RATE) / CHUNK) + 1
            start_index = max(0, len(audio_buffer) - num_chunks_needed)
            audio_segment_bytes = b''.join(list(audio_buffer)[start_index:])
            mel_chunk = None # مقدار اولیه
            if len(audio_segment_bytes) >= int(0.2 * RATE * 2):
                 mel_chunk = get_mel_chunk(audio_segment_bytes) # انتظار 4D
                 if mel_chunk is not None: mel_chunk = mel_chunk.to(DEVICE)

            # --- اجرای مدل Wav2Lip (با بررسی ابعاد) ---
            generated_faces = None
            if face_batch is not None and mel_chunk is not None and len(valid_indices) > 0:
                # ----> چاپ ابعاد برای دیباگ <----
                print(f"DEBUG: Before model call - Mel shape: {mel_chunk.shape}, Face batch shape: {face_batch.shape}")
                face_batch = face_batch.to(DEVICE)
                with torch.no_grad():
                    try:
                        # اجرای مدل
                        generated_faces_batch = model(mel_chunk, face_batch)
                        # پردازش خروجی
                        generated_faces = generated_faces_batch.squeeze(0).cpu().numpy()
                        if generated_faces is not None and generated_faces.size > 0:
                            print(
                                f"DEBUG: Model Output Stats - Min: {generated_faces.min():.4f}, Max: {generated_faces.max():.4f}, Mean: {generated_faces.mean():.4f}")
                        generated_faces = np.transpose(generated_faces, (0, 2, 3, 1))
                        # بازگردانی مقادیر [0, 1] خروجی Sigmoid به [0, 255]
                        generated_faces = np.clip(generated_faces * 255.0, 0, 255).astype(np.uint8)
                        # ذخیره در کش
                        for idx, face in zip(valid_indices, generated_faces): generated_face_cache[idx] = face
                    except RuntimeError as e: print(f"خطای Runtime مدل: {e}") # چاپ خطای دقیق مدل
                    except Exception as e: print(f"خطای اجرای مدل: {e}")
            # else:
            #      print(f"DEBUG: Skipping model execution. face_batch is None: {face_batch is None}, mel_chunk is None: {mel_chunk is None}, len(valid_indices): {len(valid_indices) if valid_indices else 0}")


            # --- آماده سازی فریم خروجی ---
            output_frame = original_frame_buffer[0].copy()
            output_bbox = bboxes_to_process[0]
            output_face_to_paste = generated_face_cache.get(0)

            # --- آماده سازی فریم خروجی ---
            # از آخرین فریم در بافر اصلی کپی بگیر (فریم متناظر با اولین چهره تولیدی)
            output_frame = original_frame_buffer[0].copy()
            output_bbox = bboxes_to_process[0]
            # چهره تولید شده متناظر با این فریم رو از کش بگیر
            output_face_to_paste = generated_face_cache.get(0)

            # ----> شروع تغییرات: ترکیب با ماسک <----
            if output_face_to_paste is not None and output_bbox is not None:
                # محاسبه مختصات bounding box در فریم اصلی
                x1, y1, x2, y2 = map(int, output_bbox)
                x1, y1 = max(x1, 0), max(y1, 0)  # اطمینان از عدم خروج از مرز بالا/چپ
                x2, y2 = min(x2, output_frame.shape[1]), min(y2, output_frame.shape[
                    0])  # اطمینان از عدم خروج از مرز پایین/راست

                # فقط اگر ابعاد Bbox معتبر است، ادامه بده
                if x1 < x2 and y1 < y2:
                    try:
                        # 1. تغییر اندازه چهره تولید شده به اندازه Bbox در فریم اصلی
                        target_h, target_w = y2 - y1, x2 - x1
                        gen_face_resized = cv2.resize(
                            output_face_to_paste,  # چهره تولیدی (مثلا 96x96)
                            (target_w, target_h),  # اندازه هدف (اندازه Bbox)
                            interpolation=cv2.INTER_LANCZOS4  # یا INTER_LINEAR
                        )

                        # 2. ایجاد ماسک برای ترکیب (فقط نیمه پایینی چهره)
                        # ماسک هم‌اندازه با چهره *اصلی* تولید شده (IMG_SIZE x IMG_SIZE)
                        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                        # تعیین ردیفی که از آن به پایین ماسک سفید شود (مثلا 55% پایین‌تر از بالا)
                        # این مقدار را می‌توانید تنظیم کنید (بین 0.5 تا 0.6 معمولا خوب است)
                        mask_start_row = int(IMG_SIZE * 0.55)
                        mask[mask_start_row:, :] = 1.0  # مقدار 1 برای ناحیه دهان

                        # اعمال محوشدگی (Gaussian Blur) به ماسک برای انتقال نرم‌تر در لبه‌ها
                        # اندازه کرنل (باید فرد باشد) و سیگما را می‌توانید تنظیم کنید
                        mask_blur_kernel_size = (15, 15)  # مثلا 11، 15 یا 21
                        mask = cv2.GaussianBlur(mask, mask_blur_kernel_size, 0)

                        # 3. تغییر اندازه ماسک به اندازه Bbox در فریم اصلی
                        mask_resized = cv2.resize(
                            mask,
                            (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR  # برای ماسک معمولا خطی بهتر است
                        )
                        # اضافه کردن یک بعد کانال به ماسک برای محاسبات بعدی (h, w) -> (h, w, 1)
                        mask_resized = mask_resized[:, :, np.newaxis]

                        # 4. گرفتن ناحیه چهره اصلی از فریم خروجی (قبل از جایگذاری)
                        original_face_roi = output_frame[y1:y2, x1:x2]

                        # 5. انجام ترکیب آلفا (Alpha Blending)
                        # تبدیل به float32 برای محاسبات دقیق‌تر
                        original_face_roi_float = original_face_roi.astype(np.float32)
                        gen_face_resized_float = gen_face_resized.astype(np.float32)

                        # فرمول ترکیب: blended = original * (1 - mask) + generated * mask
                        blended_face_float = original_face_roi_float * (1.0 - mask_resized) + \
                                             gen_face_resized_float * mask_resized

                        # 6. قرار دادن ناحیه ترکیب شده در فریم خروجی
                        # تبدیل مجدد به uint8 و اطمینان از محدوده [0, 255]
                        output_frame[y1:y2, x1:x2] = np.clip(blended_face_float, 0, 255).astype(np.uint8)

                    except cv2.error as e:
                        print(f"خطا در عملیات resize/blend/paste: {e}")
                    except Exception as e:
                        import traceback
                        print(f"خطای غیرمنتظره در blend/paste: {e}")
                        # traceback.print_exc() # برای نمایش جزئیات کامل خطا (در صورت نیاز)

            elif output_bbox is not None:  # اگر bbox هست ولی چهره‌ای تولید نشده (مثلا در ابتدای کار)
                # فقط متن را نمایش بده (بدون تغییر نسبت به کد قبلی)
                x1, y1, _, _ = map(int, output_bbox)
                cv2.putText(output_frame, "Sync?", (max(0, x1), max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

            # ----> پایان تغییرات <----

            # --- پاک کردن بافرها و کش (بدون تغییر نسبت به کد قبلی) ---
            frame_buffer.popleft();
            bbox_buffer.popleft();
            original_frame_buffer.popleft()
            new_cache = {};
            for k, v in generated_face_cache.items():
                if k > 0: new_cache[k - 1] = v
            generated_face_cache = new_cache

            process_end_time = time.time()
            processing_times.append(process_end_time - process_start_time)

        else:  # اگر بافر پر نیست (بدون تغییر نسبت به کد قبلی)
            output_frame = frame.copy()
            cv2.putText(output_frame, "Buffering...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # --- نمایش FPS و اطلاعات ---
        display_counter += 1; current_time = time.time()
        elapsed_display_time = current_time - last_display_time
        if elapsed_display_time > 1.0:
            fps = display_counter / elapsed_display_time
            avg_proc_time_ms = np.mean(processing_times) * 1000 if processing_times else 0
            latency_s = WAV2LIP_BATCH_SIZE / actual_fps if actual_fps > 0 else 0
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Proc: {avg_proc_time_ms:.1f} ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Latency ~ {latency_s:.2f} s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            display_counter = 0; last_display_time = current_time

        # --- نمایش فریم نهایی ---
        cv2.imshow('Real-time Lip Sync (Wav2Lip)', output_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("کلید 'q' فشرده شد."); stop_event.set(); break

    cap.release(); cv2.destroyAllWindows(); face_detection.close()
    print("پایان پردازش ویدیو.")
# --- /پایان تابع پردازش ویدیو ---


# --- اجرای اصلی برنامه (اصلاح شده برای خطای ابعاد و flag موفقیت) ---
if __name__ == "__main__":
    # --- 1. انتخاب دستگاه صدا ---
    # ... (کد انتخاب دستگاه مثل قبل) ...
    print("DEBUG: Script execution started.", flush=True) # Keep this
    available_indices = list_audio_input_devices()
    print("DEBUG: Finished list_audio_input_devices.", flush=True) # Keep this
    INPUT_DEVICE_INDEX = None
    if not available_indices: print("دستگاه ورودی یافت نشد."); sys.exit(1)
    if len(available_indices) == 1: INPUT_DEVICE_INDEX = available_indices[0]; print(f"استفاده خودکار از Index: {INPUT_DEVICE_INDEX}")
    else:
         while INPUT_DEVICE_INDEX is None:
             try: selected = input(f"Index میکروفون را وارد کنید {available_indices}: "); candidate_index = int(selected)
             except ValueError: print("خطا: لطفا عدد وارد کنید."); continue
             except (EOFError, KeyboardInterrupt): print("\nانتخاب لغو شد."); sys.exit(0)
             if candidate_index in available_indices: INPUT_DEVICE_INDEX = candidate_index
             else: print(f"خطا: Index نامعتبر.")
    print(f"استفاده از دستگاه صوتی Index: {INPUT_DEVICE_INDEX}")

    # --- 2. بارگذاری مدل Wav2Lip ---
    # ... (کد بارگذاری مدل مثل قبل) ...
    print(f"درحال بارگذاری مدل Wav2Lip از {WAV2LIP_MODEL_PATH}...")
    model = None
    if not os.path.exists(WAV2LIP_MODEL_PATH): print(f"خطا: فایل مدل '{WAV2LIP_MODEL_PATH}' یافت نشد!"); sys.exit(1)
    try:
        model = Wav2Lip()
        checkpoint = torch.load(WAV2LIP_MODEL_PATH, map_location=DEVICE)
        if "state_dict" in checkpoint: s = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict): s = checkpoint
        else: print("خطا: فرمت Checkpoint نامشخص."); sys.exit(1)
        new_s = {};
        for k, v in s.items(): new_s[k.replace('module.', '', 1)] = v
        model.load_state_dict(new_s)
        print("مدل با موفقیت بارگذاری شد.")
    except Exception as e: print(f"خطای بارگذاری مدل: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    model = model.to(DEVICE); model.eval()

    # --- 3. راه‌اندازی تردها ---
    print("راه اندازی ترد ضبط صدا..."); audio_thread = threading.Thread(target=record_audio, args=(INPUT_DEVICE_INDEX, exit_event), daemon=True)
    audio_thread.start(); time.sleep(1)

    # --- 4. اجرای پردازش ویدیو در ترد اصلی ---
    video_processing_successful = True # فرض اولیه موفقیت
    try:
        # Warm-up (با بررسی ابعاد)
        print("Warm-up مدل (اختیاری)...")
        try:
            # ایجاد داده های ساختگی با ابعاد صحیح مورد انتظار
            dummy_mel = torch.randn(1, 1, 80, MEL_STEP_SIZE, device=DEVICE) # 4D
            dummy_face = torch.randn(1, WAV2LIP_BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE) # 5D
            print(f"DEBUG: Warmup shapes - Mel: {dummy_mel.shape}, Face: {dummy_face.shape}")
            with torch.no_grad():
                 _ = model(dummy_mel, dummy_face)
            print("Warm-up انجام شد.")
        except Exception as warmup_e:
             print(f"خطا در Warm-up: {warmup_e}. ادامه بدون warm-up...")
             # traceback.print_exc() # برای دیباگ بیشتر

        # شروع پردازش اصلی
        process_video(exit_event, model)

    except KeyboardInterrupt:
        print("\nدرخواست توقف..."); # اینجا موفقیت‌آمیز نیست لزوما
        video_processing_successful = False # کاربر متوقف کرده
    except Exception as e:
        import traceback
        print(f"\nخطای پیش بینی نشده در حلقه اصلی: {e}")
        traceback.print_exc()
        video_processing_successful = False # ---> تنظیم فلگ خطا <---
    finally:
        print("درحال تمیزکاری و خروج...")
        exit_event.set()
        if 'audio_thread' in locals() and audio_thread.is_alive():
             print("انتظار برای ترد صدا..."); audio_thread.join(timeout=2.0)
             if audio_thread.is_alive(): print("هشدار: ترد صدا خاتمه نیافت.")
        # ---> اصلاح پیام پایانی <---
        print(f"برنامه خاتمه یافت {'با موفقیت نسبی (ممکن است خطا رخ داده باشد)' if video_processing_successful else 'با خطا'}.")
# --- END OF FILE lip_sync_v2_fixed.py ---