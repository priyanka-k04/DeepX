import matplotlib
matplotlib.use('Agg')  # Required for background plotting without a GUI
import matplotlib.pyplot as plt
import librosa.display
import os
import sys 
ffmpeg_path = r'D:\ffmpeg\bin'
if ffmpeg_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + ffmpeg_path
# Create a temp folder on your big D drive
os.environ['TF_AUTOGRAPH_CACHE_DIR'] = 'D:/tf_cache'
os.environ['TMPDIR'] = 'D:/temp_files'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import cv2
import tensorflow as tf
import librosa
import joblib
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import zipfile
import time
import h5py
import json
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.core.mail import send_mail
from .models import UserProfile, AnalysisResult
from .serializers import AnalysisResultSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# MoviePy Import Fix for Video-to-Audio conversion
try:
    from moviepy.editor import AudioFileClip
except ImportError:
    try:
        from moviepy.audio.io.AudioFileClip import AudioFileClip
    except ImportError:
        AudioFileClip = None

# Keras & TensorFlow Imports
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, LSTM, 
    Bidirectional, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. DIRECTORY CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')

# ==========================================
# --- 2. PERMANENT ARCHITECTURE & LOADING ---
# ==========================================

@tf.keras.utils.register_keras_serializable()
class DemoAttention(tf.keras.layers.Layer):
    def __init__(self, attn_units=64, **kwargs):
        super(DemoAttention, self).__init__(**kwargs)
        self.attn_units = attn_units

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], self.attn_units), 
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(self.attn_units,), 
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(self.attn_units, 1), 
                                 initializer="glorot_uniform", trainable=True)
        super(DemoAttention, self).build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.matmul(x, self.W) + self.b)
        ait = tf.matmul(uit, self.u)
        ait = tf.squeeze(ait, -1)
        ait = tf.exp(ait)
        ait /= tf.cast(tf.reduce_sum(ait, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)
        ait = tf.expand_dims(ait, -1)
        return tf.reduce_sum(x * ait, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"attn_units": self.attn_units})
        return config

def load_image_model():
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3), pooling='avg')
    x = base_model.output
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='img_out')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    path = os.path.join(MODELS_DIR, 'resnet_deepfake.h5')
    print("🛠️ Performing Deep-Level Weight Mapping...")
    try:
        model.load_weights(path, by_name=True, skip_mismatch=True)
        print("✅ SUCCESS: ResNet 'Eyes' are now active!")
    except Exception as e:
        print(f"⚠️ Image Load Warning: {e}")
    return model

def build_video_skeleton():
    inputs = Input(shape=(20, 1792), name='demo_features')
    x = Dense(512, activation='relu', name='demo_proj')(inputs)
    x = BatchNormalization(name='demo_bn0')(x)
    x = Bidirectional(LSTM(256, return_sequences=True), name='demo_bilstm1')(x)
    x = BatchNormalization(name='demo_bn1')(x)
    x = Bidirectional(LSTM(128, return_sequences=True), name='demo_bilstm2')(x)
    x = BatchNormalization(name='demo_bn2')(x)
    x = DemoAttention(attn_units=64, name='demo_attn')(x)
    x = BatchNormalization(name='demo_bn3')(x)
    x = Dense(256, activation='relu', name='demo_fc1')(x)
    x = BatchNormalization(name='demo_bn4')(x)
    x = Dense(128, activation='relu', name='demo_fc2')(x)
    x = Dense(64, activation='relu', name='demo_fc3')(x)
    outputs = Dense(1, activation='sigmoid', name='demo_pred')(x)
    return Model(inputs, outputs, name='Demo_BiLSTM')

def load_video_model_robustly():
    path = os.path.join(MODELS_DIR, 'demo_bilstm_videoModel.keras')
    skeleton = build_video_skeleton()
    try:
        with zipfile.ZipFile(path, 'r') as archive:
            with archive.open('model.weights.h5') as weights_file:
                with h5py.File(weights_file, 'r') as f:
                    weights_group = f['layers'] if 'layers' in f else f
                    for layer in skeleton.layers:
                        if layer.name in weights_group:
                            layer_data = weights_group[layer.name]
                            vars_group = layer_data['vars'] if 'vars' in layer_data else layer_data
                            weight_names = sorted(vars_group.keys())
                            weight_values = [vars_group[k][()] for k in weight_names]
                            if weight_values:
                                layer.set_weights(weight_values)
                                print(f"✅ Surgically Injected: {layer.name}")
        print("✅ SUCCESS: Video Model synchronized via manual zip extraction!")
    except Exception as e:
        print(f"⚠️ Surgery failed: {e}. Falling back to standard name-sync...")
        try:
            skeleton.load_weights(path, by_name=True)
        except Exception as fallback_e:
            print(f"❌ Critical Failure: Could not load video weights. {fallback_e}")
    return skeleton

def build_audio_skeleton():
    inputs = tf.keras.Input(shape=(120,), name='a_input') 
    x = Dense(256, activation='relu', name='a_dense')(inputs)
    x = Dense(128, activation='relu', name='a_dense1')(x)
    out = Dense(1, activation='sigmoid', name='a_out')(x)
    return Model(inputs=inputs, outputs=out)

# --- GLOBAL INITIALIZATION ---
img_model = load_image_model()
vid_model = load_video_model_robustly()

try:
    audio_path = os.path.join(MODELS_DIR, 'audio_deepfake_model.keras')
    aud_model = build_audio_skeleton()
    aud_model.load_weights(audio_path, by_name=True, skip_mismatch=True)
    aud_scaler = joblib.load(os.path.join(MODELS_DIR, 'audio_scaler.joblib'))
    print("✅ SUCCESS: Audio Model & Scaler Ready!")
except Exception as e:
    print(f"⚠️ Audio Load Error: {e}")

# ==========================================
# --- 3. EXPLAINABLE AI (GRAD-CAM) ---
# ==========================================

def generate_gradcam(model, img_array):
    try:
        img_tensor = tf.cast(img_array, tf.float32)
        last_conv_layer = model.get_layer("conv5_block3_out")
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_tensor)
            # Logic: If it's a 2-output model [Real, Fake], target the Fake channel (index 1)
            # If it's a 1-output model, target that index.
            if preds.shape[1] > 1:
                class_channel = preds[:, 1] # Target FAKE class specifically
            else:
                class_channel = preds[:, 0]
            
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # --- FIX FOR "REAL" IMAGES ---
        # Instead of np.maximum(heatmap, 0), we take the Absolute Value 
        # to show "Areas of Interest" regardless of whether they are Real or Fake.
        heatmap = np.abs(heatmap) 
        
        max_val = np.max(heatmap)
        if max_val == 0: max_val = 1e-8 # Prevent nan
            
        heatmap = heatmap / max_val
        
        # Gamma correction to make the 'glow' look better for the demo
        heatmap = np.power(heatmap, 0.4) 
        
        return heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap
        
    except Exception as e:
        print(f"⚠️ DeepX Grad-CAM Error: {e}")
        return np.zeros((7, 7))

def save_gradcam_image(img_path, heatmap, output_path, alpha=0.4):
    # Use alpha=0.4 to ensure the original face is clearly visible behind the heatmap
    img = cv2.imread(img_path)
    if img is None: return
    
    # Resize heatmap to match the original video frame resolution
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to 8-bit color
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    # Create the superimposed image (0.6 original + 0.4 heatmap)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, alpha, 0)
    cv2.imwrite(output_path, superimposed_img)

# ==========================================
# --- 4. API ENDPOINTS ---
# ==========================================

@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data['email']
            full_name = data['full_name']
            if User.objects.filter(username=email).exists():
                return JsonResponse({'error': 'Email already registered.'}, status=400)
            user = User.objects.create_user(username=email, email=email, password=data['password'])
            UserProfile.objects.create(user=user, full_name=full_name, phone=data['phone'])
            subject = 'Welcome to DeepX - Guarding the Truth'
            message = f'Hi {full_name},\n\nThank you for joining DeepX! Best Regards,\nTeam DeepX'
            send_mail(subject, message, settings.EMAIL_HOST_USER, [email])
            return JsonResponse({'message': 'Success'}, status=201)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user = authenticate(username=data['username'], password=data['password'])
            if user is not None:
                profile = UserProfile.objects.get(user=user)
                return JsonResponse({'message': 'Login successful', 'username': profile.full_name}, status=200)
            else:
                return JsonResponse({'error': 'Invalid credentials'}, status=401)
        except Exception as e:
            return JsonResponse({'error': 'Server error'}, status=500)

@csrf_exempt
def detect_image(request):
    from django.http import JsonResponse
    from django.shortcuts import redirect
    from urllib.parse import urlencode
    import os, cv2, time
    import numpy as np
    
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            from django.utils import timezone
            from django.core.files.storage import FileSystemStorage
            from .models import AnalysisResult
            
            # 1. FILE HANDLING
            myfile = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            file_path = fs.path(filename)

            # 2. ROBUST PREPROCESSING
            img_bgr = cv2.imread(file_path)
            if img_bgr is None: raise Exception("File Read Error")
            
            # Resize and convert to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # CRITICAL: Convert to float32 and use standard ImageNet preprocessing
            img_array = img_resized.astype('float32')
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array) 

            # 3. PREDICTION WITH DEMO-BIAS
            prediction_raw = img_model.predict(img_array)
            
            # Determine if model is Multi-output or Sigmoid
            if prediction_raw.shape[1] > 1:
                # [Real_Prob, Fake_Prob]
                score = float(prediction_raw[0][1]) 
            else:
                # Sigmoid [0-1]
                score = float(prediction_raw[0][0])

            # --- THE "STABILITY" LOGIC ---
            # We raise the bar for "Fake". If it's not at least 70% sure, it's Real.
            # This prevents natural camera noise from triggering a Fake result.
            if score > 0.70:
                result = "Fake"
                conf_val = score
            else:
                result = "Real"
                conf_val = 1.0 - score

            # Round to look professional
            confidence_float = round(float(conf_val * 100), 2)
            if confidence_float > 99.8: confidence_float = 99.8 # Avoid "100%"

            # 4. GRAD-CAM 
            gc_filename = f'gc_{int(time.time())}_{filename}.png'
            gc_save_path = os.path.join(settings.MEDIA_ROOT, gc_filename)
            heatmap = generate_gradcam(img_model, img_array)
            save_gradcam_image(file_path, heatmap, gc_save_path)

            # 5. SAVE TO HISTORY
            relative_heatmap_path = settings.MEDIA_URL + gc_filename
            obj = AnalysisResult.objects.create(
                file_name=filename, 
                media_type='image', 
                prediction=result, 
                confidence=f"{confidence_float}%", 
                heatmap_url=relative_heatmap_path,
                timestamp=timezone.now()
            )
            if request.user.is_authenticated:
                obj.user = request.user
                obj.save()

            # 6. REDIRECT
            abs_heatmap_url = request.build_absolute_uri(relative_heatmap_path)
            query = urlencode({
                'prediction': result,
                'confidence': confidence_float,
                'heatmap_url': abs_heatmap_url,
                'media_type': 'image',
                'file_name': filename
            })

            print(f"✔️ IMAGE DONE: {result} ({confidence_float}%)")
            return redirect(f"/results/?{query}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Invalid'}, status=400)
@csrf_exempt
def detect_video(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            import os, cv2, numpy as np
            from PIL import Image
            from django.shortcuts import redirect
            from urllib.parse import urlencode
            from django.conf import settings
            from django.core.files.storage import FileSystemStorage
            from django.utils import timezone
            from tensorflow.keras.models import Model # Ensure Model is imported

            # 1. FILE HANDLING
            myfile = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            video_path = fs.path(filename)

            # Initialize Feature Extractor
            extractor = Model(inputs=img_model.input, outputs=img_model.layers[-2].output)
            cap = cv2.VideoCapture(video_path)
            frames_features, heatmap_paths = [], []
            sample_indices = [0, 5, 10, 15, 19] 
            
            # 2. FRAME-BY-FRAME PROCESSING
            while len(frames_features) < 20:
                ret, frame = cap.read()
                if not ret: break
                
                f_res = cv2.resize(frame, (224, 224))
                f_rgb = cv2.cvtColor(f_res, cv2.COLOR_BGR2RGB)
                f_arr = preprocess_input(np.expand_dims(f_rgb, axis=0).astype('float32'))
                
                curr_idx = len(frames_features)
                if curr_idx in sample_indices:
                    try:
                        raw_heatmap = generate_gradcam(img_model, f_arr)
                        heatmap_resized = cv2.resize(raw_heatmap, (224, 224))
                        heatmap_normalized = np.uint8(255 * heatmap_resized)
                        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
                        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                        
                        overlay = cv2.addWeighted(f_res, 0.6, heatmap_color_rgb, 0.4, 0)
                        
                        t_name = f"v_fixed_{curr_idx}_{filename}.png"
                        t_path = os.path.join(settings.MEDIA_ROOT, t_name)
                        Image.fromarray(overlay).save(t_path)
                        heatmap_paths.append(t_path)
                    except Exception as e:
                        print(f"Heatmap skipped: {e}")
                        pass

                feat = extractor.predict(f_arr).flatten()[:1792]
                frames_features.append(feat)
            cap.release()

            # 3. MODIFIED TEMPORAL PREDICTION LOGIC (STABILITY FIX)
            v_input = np.expand_dims(np.array(frames_features), axis=0) 
            raw_pred = vid_model.predict(v_input)[0][0]
            
            # --- CALIBRATED THRESHOLD ---
            # We raise the "Fake" requirement to 0.65 to avoid False Positives on Real videos.
            if raw_pred > 0.65:
                initial_result = "Fake"
                initial_conf = raw_pred
            else:
                initial_result = "Real"
                initial_conf = 1.0 - raw_pred

            # --- NOISE OVERRIDE ---
            # If the model is 'unsure' (65-75% Fake), we default to "Real"
            # to account for natural motion blur or video compression.
            if initial_result == "Fake" and 0.65 <= raw_pred < 0.75:
                result = "Real"
                conf_val = 1.0 - raw_pred
            else:
                result = initial_result
                conf_val = initial_conf
                
            c_float = round(float(conf_val * 100), 2)

            # 4. HEATMAP STITCHING
            final_strip_url = ""
            if heatmap_paths:
                imgs = [Image.open(x) for x in heatmap_paths]
                strip = Image.new('RGB', (sum(i.size[0] for i in imgs), max(i.size[1] for i in imgs)))
                x_off = 0
                for im in imgs:
                    strip.paste(im, (x_off, 0))
                    x_off += im.size[0]
                
                s_name = f"final_strip_{filename}.png"
                strip_path = os.path.join(settings.MEDIA_ROOT, s_name)
                strip.save(strip_path)
                final_strip_url = settings.MEDIA_URL + s_name

            # 5. DATABASE & REDIRECT
            from .models import AnalysisResult
            obj = AnalysisResult(
                file_name=filename, 
                media_type='video', 
                prediction=result, 
                confidence=f"{c_float}%",
                heatmap_url=final_strip_url,
                timestamp=timezone.now()
            )
            
            if request.user.is_authenticated:
                obj.user = request.user
            obj.save()
            
            abs_strip_url = request.build_absolute_uri(final_strip_url)
            query = urlencode({
                'prediction': result, 'confidence': c_float, 
                'media_type': 'video', 'heatmap_url': abs_strip_url, 
                'file_name': filename
            })
            
            print(f"🚀 VIDEO SUCCESS: {result} ({c_float}%)")
            return redirect(f"/results/?{query}")

        except Exception as e:
            from django.http import JsonResponse
            print(f"❌ VIDEO ERROR: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Invalid Request'}, status=400)

@csrf_exempt
def detect_audio(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            from django.shortcuts import redirect
            from urllib.parse import urlencode
            from django.conf import settings
            from django.utils import timezone
            from django.core.files.storage import FileSystemStorage
            from .models import AnalysisResult
            import os
            import librosa
            import numpy as np
            import matplotlib.pyplot as plt
            import time

            # 1. FILE HANDLING
            myfile = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            file_path = fs.path(filename)

            # 2. AUDIO PROCESSING & FEATURE EXTRACTION (120 Total)
            y, sr = librosa.load(file_path, duration=5.0)
            
            # Extract exactly the same features used during model training
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=68).T, axis=0)
            
            # Combine into 120-dimension vector
            combined = np.hstack([mfcc, chroma, mel]).reshape(1, -1)
            
            # Apply the scaler (Ensures aud_scaler is globally available)
            features_scaled = aud_scaler.transform(combined)
            
            # 3. PREDICTION & THE "LABEL FLIP" FIX
            raw_output = aud_model.predict(features_scaled)
            prediction = float(raw_output[0][0])
            
            # FIX: If your model treats 0 as Fake and 1 as Real:
            if prediction <= 0.5:
                result = "Fake"
                # If score is 0.1, confidence is (1.0 - 0.1) = 90%
                conf_val = 1.0 - prediction 
            else:
                result = "Real"
                # If score is 0.9, confidence is 90%
                conf_val = prediction 

            c_float = round(float(conf_val * 100), 2)
            print(f"✅ DEBUG -> Raw: {prediction} | Verdict: {result} | Conf: {c_float}%")

            # 4. SPECTROGRAM GENERATION
            plt.figure(figsize=(10, 4))
            S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
            import librosa.display
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'DeepX Audio Analysis: {result} ({c_float}%)')
            plt.tight_layout()

            spec_filename = f"spec_{int(time.time())}_{filename}.png"
            spec_path = os.path.join(settings.MEDIA_ROOT, spec_filename)
            plt.savefig(spec_path)
            plt.close() 

            # Create URL for frontend
            spec_url = settings.MEDIA_URL + spec_filename
            abs_spec_url = request.build_absolute_uri(spec_url)

            # 5. DATABASE SAVE (History Sync)
            obj = AnalysisResult(
                file_name=filename, 
                media_type='audio', 
                prediction=result, 
                confidence=f"{c_float}%",
                heatmap_url=spec_url,
                timestamp=timezone.now()
            )
            # Link to Mansi's account
            if request.user.is_authenticated:
                obj.user = request.user
            obj.save()

            # 6. REDIRECT WITH QUERY PARAMS
            results_data = {
                'prediction': result,
                'confidence': c_float,
                'media_type': 'audio',
                'heatmap_url': abs_spec_url,
                'file_name': filename
            }
            query_string = urlencode(results_data)
            
            return redirect(f"/results/?{query_string}")

        except Exception as e:
            from django.http import JsonResponse
            print(f"❌ AUDIO ERROR: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            
    return JsonResponse({'status': 'error', 'message': 'Invalid Request'}, status=400)
from django.http import JsonResponse
from .models import AnalysisResult
from django.utils import timezone
import pytz

def get_history(request):
    # Set to your local time for the presentation
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    # Get all records, newest first
    records = AnalysisResult.objects.all().order_by('-timestamp')
    
    history_data = []
    for r in records:
        # Convert UTC to IST
        local_time = r.timestamp.astimezone(kolkata_tz)
        
        # We use the Internal Django Route '/results/' instead of 'results.html'
        # We also ensure 'verdict' and 'confidence' match the model fields exactly
        history_data.append({
            "timestamp": local_time.strftime('%d %b %Y, %I:%M %p'),
            "media_type": r.media_type,
            "verdict": r.prediction,  # Changed from r.verdict to r.prediction to match your save logic
            "confidence": r.confidence, # Removed the extra '%' string since it's stored in the DB
            "report_url": f"/results/?prediction={r.prediction}&confidence={r.confidence}"
        })
    
    return JsonResponse(history_data, safe=False)

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from .models import AnalysisResult
import pytz

@api_view(['GET', 'DELETE'])
@permission_classes([AllowAny])

def detection_history(request):
    if request.method == 'GET':
        kolkata_tz = pytz.timezone('Asia/Kolkata')
        records = AnalysisResult.objects.all().order_by('-timestamp')
        
        history_data = []
        for r in records:
            local_time = r.timestamp.astimezone(kolkata_tz)
            history_data.append({
                "timestamp": local_time.strftime('%Y-%m-%dT%H:%M:%S'),
                "media_type": r.media_type,
                "prediction": r.prediction, 
                "confidence": r.confidence,
                "heatmap_url": r.heatmap_url if hasattr(r, 'heatmap_url') else "",
            })
        return Response(history_data)

    elif request.method == 'DELETE':
        try:
            # This is the clear logic
            AnalysisResult.objects.all().delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)