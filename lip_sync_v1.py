# --- START OF LipSync Project - Step 5 Completed (tf.function Fixed) ---

import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import queue
import librosa
import librosa.feature # Explicitly import feature submodule
import sys
import time
import os # اضافه شده برای تنظیم متغیر محیطی

# راه حل موقت برای خطای OpenMP در ویندوز
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf # وارد کردن TensorFlow
from tensorflow import keras # وارد کردن Keras

# غیرفعال کردن لاگ های زیاد TensorFlow (اختیاری)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # یا '3'

# --- بخش ۱.الف: شناسایی و لیست کردن دستگاه‌های ورودی صدا ---
def list_audio_input_devices():
    p = pyaudio.PyAudio(); info = p.get_host_api_info_by_index(0); numdevices = info.get('deviceCount', 0)
    print("-" * 60); print("دستگاه‌های ورودی صوتی یافت شده:"); print("-" * 60)
    found_input_device = False; available_indices = []
    for i in range(0, numdevices):
        try:
             device_info = p.get_device_info_by_host_api_device_index(0, i)
             if device_info.get('maxInputChannels') > 0:
                 found_input_device = True; available_indices.append(i)
                 try: device_name = device_info.get('name', b'Unknown').decode('utf-8', errors='replace')
                 except UnicodeDecodeError:
                     try: device_name = device_info.get('name', b'Unknown').decode('latin-1', errors='replace')
                     except Exception: device_name = f"Unknown (Index {i})"
                 except Exception: device_name = f"Unknown (Index {i})"
                 print(f"  Index {i}: {device_name}")
        except Exception as e: print(f"  خطا در دریافت اطلاعات دستگاه Index {i}: {e}")
    if not found_input_device: print("هیچ دستگاه ورودی صوتی فعالی یافت نشد.")
    print("-" * 60)
    try: p.terminate()
    except Exception as e: print(f"خطا هنگام بستن PyAudio: {e}")
    return available_indices
# --- پایان بخش ۱.الف ---

# --- تنظیمات ---
WEBCAM_WIDTH = 640; WEBCAM_HEIGHT = 480
FORMAT = pyaudio.paInt16; CHANNELS = 1; RATE = 44100; CHUNK = 1024; N_MFCC = 13

# پارامترهای شکل لب
MIN_OPENNESS = 0.0; MAX_OPENNESS = 1.0
MIN_WIDTH_SCALE = 0.6; MAX_WIDTH_SCALE = 1.4; NEUTRAL_WIDTH_SCALE = 1.0

INPUT_DEVICE_INDEX = None
audio_queue = queue.Queue(maxsize=5)
exit_event = threading.Event()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

UPPER_LIP_POINTS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_POINTS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LIP_DISTANCE_TOP_INDEX = 13; LIP_DISTANCE_BOTTOM_INDEX = 14
LIP_CORNER_LEFT_INDEX = 78; LIP_CORNER_RIGHT_INDEX = 308

VISEME_CLOSED_DISTANCE = 3
VISEME_OPEN_DISTANCE = 25

# هموارسازی
feature_smoothing_factor = 0.7
target_smoothing_factor = 0.8

EXPECTED_FEATURE_LENGTH = 1 + (N_MFCC * 3) + 1 + 1 # 54
smoothed_feature_vector = np.zeros(EXPECTED_FEATURE_LENGTH, dtype=np.float32)

# --- تعریف و بارگذاری مدل پایه MLP (گام ۵.پ) ---
def create_basic_mlp_model(input_shape, output_units=2):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,), dtype=tf.float32),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_units, activation='sigmoid', dtype=tf.float32)
    ])
    print("مدل MLP پایه ایجاد شد (بدون آموزش).")
    return model

basic_model = create_basic_mlp_model(input_shape=EXPECTED_FEATURE_LENGTH, output_units=2)
# --- /پایان بخش جدید ---

# --- تابع ضبط صوت (بدون تغییر از گام ۴) ---
def record_audio(device_index, stop_event):
    p = pyaudio.PyAudio(); stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=device_index)
        print(f"🎤 ضبط صوت از Index {device_index} آغاز شد...")
    except IOError as e: print(f"خطا باز کردن stream: {e}"); p.terminate(); stop_event.set(); return

    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if stop_event.is_set(): break
            try: audio_queue.put(data, block=True, timeout=0.01)
            except queue.Full:
                 try: audio_queue.get_nowait(); audio_queue.put_nowait(data)
                 except queue.Empty: pass
                 except queue.Full: pass
        except IOError as e:
            if e.errno == -9988: print("Stream صدا بسته شد.")
            elif e.errno == -9981: time.sleep(0.005); continue
            else: print(f"خطای IO صدا: {e}")
            stop_event.set(); break
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
# --- /پایان تابع ضبط صوت ---

# --- تابع استخراج ویژگی‌ها (بدون تغییر از گام ۴) ---
def extract_audio_features(audio_data, sample_rate=RATE, n_mfcc=N_MFCC):
    global EXPECTED_FEATURE_LENGTH
    rms = 0; mfccs_mean = np.zeros(n_mfcc); delta_mfccs_mean = np.zeros(n_mfcc)
    delta2_mfccs_mean = np.zeros(n_mfcc); zcr_mean = 0; centroid_mean = 0
    feature_vector = np.zeros(EXPECTED_FEATURE_LENGTH, dtype=np.float32)
    try:
        audio_signal = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        try: rms = np.sqrt(np.mean(np.square(audio_signal))) * 10000
        except Exception: pass
        mfccs = np.zeros((n_mfcc, 1), dtype=np.float32); delta_mfccs = np.zeros((n_mfcc, 1), dtype=np.float32)
        try:
            mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=CHUNK, hop_length=CHUNK*2)
            if mfccs.shape[1] > 0: mfccs_mean = np.mean(mfccs, axis=1)
        except Exception: pass
        try:
             if mfccs.shape[1] >= 3 :
                 delta_mfccs = librosa.feature.delta(mfccs, width=3)
                 delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
        except Exception: pass
        try:
             if delta_mfccs.shape[1] >= 3:
                 delta2_mfccs = librosa.feature.delta(delta_mfccs, width=3)
                 delta2_mfccs_mean = np.mean(delta2_mfccs, axis=1)
        except Exception: pass
        try:
             zcr = librosa.feature.zero_crossing_rate(y=audio_signal, frame_length=CHUNK, hop_length=CHUNK*2)[0]
             if len(zcr) > 0: zcr_mean = np.mean(zcr)
        except Exception: pass
        try:
             cent = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate, n_fft=CHUNK, hop_length=CHUNK*2)[0]
             if len(cent) > 0: centroid_mean = np.mean(cent)
        except Exception: pass
        feature_vector = np.concatenate(([rms], mfccs_mean, delta_mfccs_mean, delta2_mfccs_mean, [zcr_mean], [centroid_mean])).astype(np.float32)
        current_length = len(feature_vector)
        if current_length != EXPECTED_FEATURE_LENGTH:
             if current_length < EXPECTED_FEATURE_LENGTH: feature_vector = np.concatenate((feature_vector, np.zeros(EXPECTED_FEATURE_LENGTH - current_length, dtype=np.float32)))
             else: feature_vector = feature_vector[:EXPECTED_FEATURE_LENGTH]
    except ImportError: print("هشدار: librosa نصب نیست."); #... (RMS failsafe)
    except Exception as e: print(f"خطای کلی استخراج: {e}"); #... (RMS failsafe)
    if not np.all(np.isfinite(feature_vector)): feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feature_vector
# --- /پایان تابع استخراج ویژگی‌ها ---


# --- تابع پردازش با مدل AI (حذف موقت tf.function) ---

# @tf.function(...) # ----> این خط کامنت شود
# def run_model_inference(model, inp, training):
#      """اجرای استنتاج مدل با استفاده از گراف tf.function."""
#      return model(inp, training=training)

def get_shape_params_from_model(feature_vector, model):
    global MIN_OPENNESS, MAX_OPENNESS, MIN_WIDTH_SCALE, MAX_WIDTH_SCALE, NEUTRAL_WIDTH_SCALE
    try:
        # 5.ب: آماده سازی ورودی
        model_input = np.expand_dims(feature_vector.astype(np.float32), axis=0)

        # 5.ت: اجرای استنتاج مدل (فراخوانی مستقیم)
        # raw_output = run_model_inference(model, tf.constant(model_input), tf.constant(False)) # کامنت شود
        raw_output = model(tf.constant(model_input), training=False) # <--- فراخوانی مستقیم مدل

        # 5.ث: پس پردازش خروجی
        predicted_params = raw_output[0].numpy()
        openness = MIN_OPENNESS + predicted_params[0] * (MAX_OPENNESS - MIN_OPENNESS)
        width = MIN_WIDTH_SCALE + predicted_params[1] * (MAX_WIDTH_SCALE - MIN_WIDTH_SCALE)
        openness = np.clip(openness, MIN_OPENNESS, MAX_OPENNESS)
        width = np.clip(width, MIN_WIDTH_SCALE, MAX_WIDTH_SCALE)
        return {'openness': float(openness), 'width': float(width)}
    except Exception as e:
        print(f"خطا در استنتاج مدل: {e}") # خطا را چاپ کن تا ببینیم چیست
        return {'openness': MIN_OPENNESS, 'width': NEUTRAL_WIDTH_SCALE}
# --- /پایان تابع پردازش با مدل AI ---


# --- تابع تنظیم لندمارک‌ها (بدون تغییر از گام ۴) ---
def adjust_lip_landmarks(landmarks, openness, width_scale, frame_width, frame_height):
    try:
        top_lip_ref_pt = landmarks.landmark[LIP_DISTANCE_TOP_INDEX]; bottom_lip_ref_pt = landmarks.landmark[LIP_DISTANCE_BOTTOM_INDEX]
        left_corner_pt = landmarks.landmark[LIP_CORNER_LEFT_INDEX]; right_corner_pt = landmarks.landmark[LIP_CORNER_RIGHT_INDEX]
        top_lip_center_y = top_lip_ref_pt.y * frame_height; bottom_lip_center_y = bottom_lip_ref_pt.y * frame_height
        current_lip_distance = bottom_lip_center_y - top_lip_center_y
        if current_lip_distance < 1: current_lip_distance = 1
        lip_center_y = (top_lip_center_y + bottom_lip_center_y) / 2; lip_center_x = (left_corner_pt.x + right_corner_pt.x) * frame_width / 2
        target_distance = VISEME_CLOSED_DISTANCE + openness * (VISEME_OPEN_DISTANCE - VISEME_CLOSED_DISTANCE); target_distance = max(target_distance, 0)
        scale_y = target_distance / current_lip_distance; scale_y = np.clip(scale_y, 0.1, 5.0)
        scale_x = width_scale; scale_x = np.clip(scale_x, MIN_WIDTH_SCALE - 0.1, MAX_WIDTH_SCALE + 0.1)
        new_upper_lip_points = []; new_lower_lip_points = []
        for idx in UPPER_LIP_POINTS:
            lm = landmarks.landmark[idx]; ox, oy = lm.x * frame_width, lm.y * frame_height
            dx = ox - lip_center_x; dy = oy - lip_center_y; nx, ny = int(lip_center_x + dx * scale_x), int(lip_center_y + dy * scale_y)
            new_upper_lip_points.append((np.clip(nx, 0, frame_width - 1), np.clip(ny, 0, frame_height - 1)))
        for idx in LOWER_LIP_POINTS:
            lm = landmarks.landmark[idx]; ox, oy = lm.x * frame_width, lm.y * frame_height
            dx = ox - lip_center_x; dy = oy - lip_center_y; nx, ny = int(lip_center_x + dx * scale_x), int(lip_center_y + dy * scale_y)
            new_lower_lip_points.append((np.clip(nx, 0, frame_width - 1), np.clip(ny, 0, frame_height - 1)))
        return new_upper_lip_points, new_lower_lip_points
    except Exception as e:
        original_upper = [(int(landmarks.landmark[idx].x * frame_width), int(landmarks.landmark[idx].y * frame_height)) for idx in UPPER_LIP_POINTS]
        original_lower = [(int(landmarks.landmark[idx].x * frame_width), int(landmarks.landmark[idx].y * frame_height)) for idx in LOWER_LIP_POINTS]
        return original_upper, original_lower
# --- /پایان تابع تنظیم لندمارک‌ها ---


# --- تابع پردازش ویدیو (بدون تغییر از اصلاح قبلی) ---
def process_video(stop_event):
    global smoothed_feature_vector, basic_model
    cap = cv2.VideoCapture(0);
    if not cap.isOpened(): print("خطا باز کردن وب کم."); stop_event.set(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    print("📸 پردازش تصویر آغاز شد...")
    smoothed_openness = MIN_OPENNESS; smoothed_width = NEUTRAL_WIDTH_SCALE
    frame_count = 0; start_time = time.time()

    while cap.isOpened() and not stop_event.is_set():
        audio_data = None; new_audio_data_received = False; data_read_count = 0
        try:
            while not audio_queue.empty(): last_data = audio_queue.get_nowait(); audio_data = last_data; new_audio_data_received = True; data_read_count += 1
        except queue.Empty: pass
        if data_read_count > 1:
            with audio_queue.mutex: audio_queue.queue.clear()

        if new_audio_data_received and audio_data:
            raw_feature_vector = extract_audio_features(audio_data, sample_rate=RATE, n_mfcc=N_MFCC)
            if len(raw_feature_vector) == EXPECTED_FEATURE_LENGTH:
                 clipped_raw_vector = np.clip(raw_feature_vector, -1e6, 1e6) # کلیپ کردن ورودی
                 smoothed_feature_vector = ((1 - feature_smoothing_factor) * clipped_raw_vector + feature_smoothing_factor * smoothed_feature_vector)
        else: smoothed_feature_vector *= feature_smoothing_factor

        shape_params = get_shape_params_from_model(smoothed_feature_vector, basic_model)
        target_openness = shape_params['openness']; target_width = shape_params['width']

        smoothed_openness = ((1 - target_smoothing_factor) * target_openness + target_smoothing_factor * smoothed_openness)
        smoothed_width = ((1 - target_smoothing_factor) * target_width + target_smoothing_factor * smoothed_width)

        ret, frame = cap.read()
        if not ret: print("خطا خواندن فریم."); stop_event.set(); break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame); rgb_frame.flags.writeable = True
        final_upper_lip, final_lower_lip = [], []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            final_upper_lip, final_lower_lip = adjust_lip_landmarks(landmarks=face_landmarks, openness=smoothed_openness, width_scale=smoothed_width, frame_width=WEBCAM_WIDTH, frame_height=WEBCAM_HEIGHT)
            if final_upper_lip and final_lower_lip:
                 cv2.polylines(frame, [np.array(final_upper_lip)], isClosed=False, color=(255, 255, 255), thickness=1)
                 cv2.polylines(frame, [np.array(final_lower_lip)], isClosed=False, color=(255, 255, 255), thickness=1)

        frame_count += 1; elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
             fps = frame_count / elapsed_time
             cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
             frame_count = 0; start_time = time.time()

        cv2.imshow('Real-time Lip Sync (Step 5 - Basic AI Placeholder)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("کلید 'q' فشرده شد."); stop_event.set(); break
    cap.release(); cv2.destroyAllWindows(); print("پایان پردازش ویدیو.")
# --- /پایان تابع پردازش ویدیو ---

# --- اجرای اصلی برنامه ---
if __name__ == "__main__":
    available_indices = list_audio_input_devices()
    INPUT_DEVICE_INDEX = None
    if not available_indices: print("دستگاه ورودی یافت نشد."); sys.exit(1)
    if len(available_indices) == 1: INPUT_DEVICE_INDEX = available_indices[0]; print(f"استفاده از Index: {INPUT_DEVICE_INDEX}")
    else:
         while INPUT_DEVICE_INDEX is None:
             try:
                 selected = input(f"Index میکروفون را وارد کنید: ")
                 candidate_index = int(selected)
                 if candidate_index in available_indices: INPUT_DEVICE_INDEX = candidate_index
                 else: print(f"خطا: Index نامعتبر.")
             except ValueError: print("خطا: لطفا عدد وارد کنید.")
             except EOFError: print("\nورودی قطع شد."); INPUT_DEVICE_INDEX = available_indices[0]; break
             except KeyboardInterrupt: print("\nانتخاب لغو شد."); sys.exit(0)
    print(f"استفاده از دستگاه صوتی Index: {INPUT_DEVICE_INDEX}")

    audio_thread = threading.Thread(target=record_audio, args=(INPUT_DEVICE_INDEX, exit_event))
    audio_thread.start()

    video_processing_successful = True
    try:
        # اجرای اول استنتاج برای ساخت گراف tf.function (warm-up)
        print("اجرای اولیه مدل برای ساخت گراف...")
        _ = get_shape_params_from_model(np.zeros(EXPECTED_FEATURE_LENGTH, dtype=np.float32), basic_model)
        print("Warm-up مدل انجام شد.")
        # حالا اجرای اصلی
        process_video(exit_event)
    except KeyboardInterrupt: print("\nدرخواست توقف..."); exit_event.set()
    except Exception as e: print(f"\nخطای پیش بینی نشده: {e}"); exit_event.set(); video_processing_successful = False
    finally:
        if audio_thread.is_alive():
             print("انتظار برای ترد صدا..."); audio_thread.join(timeout=5.0)
             if audio_thread.is_alive(): print("هشدار: ترد صدا خاتمه نیافت.")
        print(f"برنامه خاتمه یافت {'با موفقیت' if video_processing_successful else 'با خطا'}.")
# --- END OF LipSync Project - Step 5 Completed (tf.function Fixed) ---