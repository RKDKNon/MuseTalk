# Patch mmcv-lite: stub mmcv._ext so mmpose can import without compiled CUDA ops
import importlib, importlib.machinery, types, sys

class _StubExt(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = '<mmcv_lite_stub>'
        self.__loader__ = importlib.machinery.SourceFileLoader(name, '<mmcv_lite_stub>')
        self.__spec__ = importlib.machinery.ModuleSpec(name, self.__loader__)
        self.__package__ = 'mmcv'
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        def _stub(*args, **kwargs):
            raise NotImplementedError(f"mmcv._ext.{name} requires full mmcv with CUDA ops")
        return _stub

_orig_import = importlib.import_module
def _patched_import(name, *args, **kwargs):
    if name == 'mmcv._ext':
        if name not in sys.modules:
            sys.modules[name] = _StubExt(name)
        return sys.modules[name]
    return _orig_import(name, *args, **kwargs)
importlib.import_module = _patched_import

# Patch torch.load for PyTorch 2.6+
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ═══════════════════════════════════════════════════════════════
# Live AI Avatar Studio — Redesigned TikTok Live Dashboard
# ═══════════════════════════════════════════════════════════════
import os
import time
import json
import re
import argparse
import tempfile
import numpy as np
import cv2
import glob
import pickle
import copy
import shutil
import subprocess

import gradio as gr
import requests
import imageio

try:
    from moviepy.editor import *
except ImportError:
    from moviepy import *

from argparse import Namespace
from transformers import WhisperModel

# ═══════════════════════════════════════════════════════════════
# Configurations & Paths
# ═══════════════════════════════════════════════════════════════
ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")
SAMPLE_AVATAR_PATH = os.path.join(os.path.dirname(ProjectDir), "ai.png")

# Load API config
API_CONFIG_PATH = os.path.join(os.path.dirname(ProjectDir), "api.json")
if os.path.exists(API_CONFIG_PATH):
    try:
        with open(API_CONFIG_PATH, 'r', encoding='utf-8') as f:
            API_CONFIG = json.load(f)
    except Exception:
        API_CONFIG = {"providers": {}}
else:
    API_CONFIG = {"providers": {}}

def get_llm_choices():
    """Extract model choices from api.json"""
    choices = []
    for provider_id, provider in API_CONFIG.get("providers", {}).items():
        for model_id, model_info in provider.get("models", {}).items():
            label = f"{provider['name']} / {model_info['name']}"
            choices.append(label)
    return choices

def get_llm_config(choice_label):
    """Get API config for a selected model"""
    for provider_id, provider in API_CONFIG.get("providers", {}).items():
        for model_id, model_info in provider.get("models", {}).items():
            label = f"{provider['name']} / {model_info['name']}"
            if label == choice_label:
                api_key = provider["settings"].get("apiKey", "")
                if not api_key and provider.get("env"):
                    for env_var in provider["env"]:
                        if env_var in os.environ:
                            api_key = os.environ[env_var]
                            break
                return {
                    "base_url": provider["settings"]["baseURL"],
                    "api_key": api_key,
                    "model_id": model_info["modelID"],
                    "provider": provider["name"],
                    "model_name": model_info["name"]
                }
    return None

# Voice style presets (Thai labels → English instruct for OmniVoice)
VOICE_STYLE_MAP = {
    "👩 หญิง - เสียงสดใสไลฟ์สด (แนะนำ)": "female, young adult, high pitch, energetic",
    "👩 หญิง - เสียงหวาน นุ่มนวล": "female, moderate pitch, soft and gentle",
    "👩 หญิง - เสียงผู้ใหญ่ น่าเชื่อถือ": "female, low pitch, confident",
    "👩 หญิง - เสียงกระซิบ มีเสน่ห์": "female, whisper, alluring",
    "👨 ชาย - เสียงสดใส วัยรุ่น": "male, young adult, energetic",
    "👨 ชาย - เสียงทางการ มั่นคง": "male, moderate pitch, formal",
    "👨 ชาย - เสียงทุ้มนุ่ม อบอุ่น": "male, low pitch, warm",
    "🎙️ โคลนเสียง (อัปโหลดไฟล์เสียงต้นแบบ)": "__clone__",
}
VOICE_STYLES = list(VOICE_STYLE_MAP.keys())

# Preset templates for products
PRESET_PRODUCTS = {
    "skincare": "Apple Extract MCT Oil คุมหิว อิ่มนาน เร่งการเผาผลาญ สกัดจากแอปเปิ้ลธรรมชาติ ทานง่าย ไม่เวียนหัว\nโปรพิเศษเปิดตัวในไลฟ์: 1 แถม 1 เพียง 390 บาท (ปกติ 780.-) ส่งฟรีทั่วประเทศ มีเก็บเงินปลายทาง",
    "fashion": "เดรสคอเหลี่ยมสไตล์เกาหลี ผ้าไหมซาตินพรีเมียม นุ่มลื่น ทรงทิ้งตัวสวย ใส่แล้วดูผอมเพรียว มีครบไซส์ S-2XL\nโปรเฉพาะไลฟ์นี้: ลดเหลือ 290 บาท ซื้อ 2 ตัวแถมเข็มขัดแฟชั่นฟรี",
    "gadget": "หูฟังบลูทูธไร้สาย Wireless Pro ตัดเสียงรบกวน ANC เบสแน่น เสียงคมชัด แบตเตอรี่อึด 28 ชั่วโมง กันน้ำ IPX5\nแฟลชเซลไลฟ์สด: 490 บาท จากปกติ 1,290 บาท มีประกันศูนย์ 1 ปีเต็ม"
}

# ═══════════════════════════════════════════════════════════════
# Load MuseTalk Models
# ═══════════════════════════════════════════════════════════════
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor

print("Loading MuseTalk models...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path=os.path.join(CheckpointsDir, "musetalkV15", "unet.pth"),
    vae_type="sd-vae",
    unet_config=os.path.join(CheckpointsDir, "musetalkV15", "musetalk.json"),
    device=device
)
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
weight_dtype = unet.model.dtype
timesteps = torch.tensor([0], device=device)

audio_processor = AudioProcessor(feature_extractor_path=os.path.join(CheckpointsDir, "whisper"))
whisper = WhisperModel.from_pretrained(os.path.join(CheckpointsDir, "whisper"))
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
print("MuseTalk models loaded!")

# ═══════════════════════════════════════════════════════════════
# LLM Script Generation & Quick Test
# ═══════════════════════════════════════════════════════════════
def generate_script_llm(model_choice, system_prompt, product_info, progress=gr.Progress()):
    """Generate script using selected LLM"""
    config = get_llm_config(model_choice)
    if not config:
        return "❌ ไม่พบการตั้งค่าโมเดล AI ที่เลือก"
    if not product_info.strip():
        return "⚠️ กรุณากรอกรายละเอียดสินค้า หรือคลิกเลือกตัวอย่างสินค้าด้านบน"
    
    progress(0.1, desc="🧠 กำลังคิดสคริปต์ขายของ...")
    
    base_url = config["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        if "/v1/" not in base_url:
            base_url = base_url + "/v1" if not base_url.endswith("/v1") else base_url
    
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    
    if not system_prompt.strip():
        system_prompt = """คุณคือนักไลฟ์ขายของออนไลน์มือโปร สไตล์ TikTok Live
พูดจาสดใส ดึงดูด กระตุ้นยอดขายอย่างเป็นธรรมชาติ ภาษาไทยชัดเจน
เขียนสคริปต์สั้นกระชับ 3-4 ประโยคต่อช่วง ไม่ใส่อีโมจิเยอะเกินไป ไม่ใส่เครื่องหมาย markdown ซับซ้อน เพื่อให้ AI อ่านเสียงได้ลื่นไหล"""

    user_msg = f"สร้างสคริปต์ไลฟ์สดขายสินค้านี้ (เน้นทักทายคนดู, ชูจุดเด่นสินค้า, แจ้งโปรโมชันด่วน, ชวนกดสั่งในตะกร้า):\n\n{product_info}"
    
    payload = {
        "model": config["model_id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.75,
        "max_tokens": 1024,
        "stream": False
    }
    
    try:
        progress(0.35, desc=f"📡 เรียกใช้ AI ({config['provider']})...")
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        script = data["choices"][0]["message"]["content"].strip()
        progress(1.0, desc="✅ สร้างสคริปต์สำเร็จ!")
        return script
    except requests.exceptions.Timeout:
        return "❌ หมดเวลาเชื่อมต่อ (Timeout) กรุณาลองใหม่อีกครั้ง"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาดจาก API: {str(e)}"

def test_ai_brain(model_choice):
    """Test AI Brain connection and return compact badge info"""
    config = get_llm_config(model_choice)
    if not config:
        return "❌ ไม่พบโมเดล", ""
    
    base_url = config["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1" if "/v1/" not in base_url else base_url
    url = f"{base_url}/chat/completions"
    
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    
    payload = {
        "model": config["model_id"],
        "messages": [
            {"role": "system", "content": "ตอบเป็นภาษาไทยสั้นๆ 1-2 ประโยค"},
            {"role": "user", "content": "แนะนำครีมทาหน้าสั้นๆ 2 ประโยค"}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": False
    }
    
    try:
        start = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        elapsed = time.time() - start
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip() and len(s.strip()) > 3]
        summary = f"🟢 พร้อมใช้งาน! ({config['provider']}) • ตอบสนอง {elapsed:.2f}s • {len(sentences)} ประโยค ({len(text)} อักษร)"
        return summary, text
    except Exception as e:
        err_msg = f"🔴 เชื่อมต่อไม่สำเร็จ: {str(e)[:120]}"
        return err_msg, ""

# ═══════════════════════════════════════════════════════════════
# TTS Generation (OmniVoice)
# ═══════════════════════════════════════════════════════════════
def generate_tts(text, voice_style, clone_audio_path=None, progress=gr.Progress()):
    """Generate TTS audio using OmniVoice TTS server or CLI"""
    if not text or not text.strip():
        return None, "กรุณากรอกสคริปต์ข้อความก่อนสร้างเสียง"
        
    progress(0.15, desc="🎙️ กำลังสังเคราะห์เสียงพากย์...")
    
    instruct = VOICE_STYLE_MAP.get(voice_style, "female, young adult, high pitch, energetic")
    is_clone = (instruct == "__clone__")
    
    audio_path = os.path.join(ProjectDir, "results", "tts_output.wav")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    # 1. Try OmniVoice TTS server if running
    try:
        data_payload = {"text": text, "instruct": instruct if not is_clone else "moderate pitch", "num_step": 16}
        files = {}
        if is_clone and clone_audio_path and os.path.exists(clone_audio_path):
            files = {"ref_audio": open(clone_audio_path, "rb")}
            
        response = requests.post("http://localhost:8765/tts/generate", data=data_payload, files=files if files else None, timeout=30)
        if response.status_code == 200:
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            progress(1.0, desc="✅ สร้างเสียงพากย์สำเร็จ!")
            return audio_path, "✅ สร้างเสียงสำเร็จ (Server)"
    except Exception:
        pass
    
    # 2. Fallback to CLI
    try:
        progress(0.3, desc="🎙️ ประมวลผลผ่าน OmniVoice CLI...")
        tts_dir = os.path.join(os.path.dirname(ProjectDir), "omnivoice-tts")
        python_exe = os.path.join(tts_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable
            
        cmd = [
            python_exe,
            "-m", "omnivoice_tts.realtime",
            "--text", text,
            "--instruct", instruct if not is_clone else "moderate pitch",
            "--output", audio_path,
            "--mode", "oneshot"
        ]
        if is_clone and clone_audio_path and os.path.exists(clone_audio_path):
            cmd.extend(["--ref_audio", clone_audio_path])
            
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=tts_dir)
        if result.returncode == 0 and os.path.exists(audio_path):
            progress(1.0, desc="✅ สร้างเสียงพากย์สำเร็จ!")
            return audio_path, "✅ สร้างเสียงสำเร็จ (CLI)"
        else:
            return None, f"❌ ไม่สามารถสร้างเสียงได้: {result.stderr[:200]}"
    except Exception as e:
        return None, f"❌ ข้อผิดพลาด TTS: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# MuseTalk Lip-Sync Inference
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def lipsync_inference(media_path, audio_path, progress=gr.Progress()):
    """Run MuseTalk v1.5 lip-sync inference on image or video"""
    if not media_path:
        return None, "❌ ไม่พบไฟล์รูปภาพหรือวิดีโออวตาร"
    if not audio_path or not os.path.exists(audio_path):
        return None, "❌ ไม่พบไฟล์เสียงสำหรับ Lip-Sync"
    
    progress(0.05, desc="🎭 เริ่มต้นประมวลผลโมเดล...")
    
    args_dict = {
        "result_dir": os.path.join(ProjectDir, "results", "studio"),
        "fps": 25,
        "batch_size": 8,
        "output_vid_name": None,
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",
        "extra_margin": 10,
        "parsing_mode": "jaw",
        "left_cheek_width": 90,
        "right_cheek_width": 90,
    }
    args = Namespace(**args_dict)
    os.makedirs(args.result_dir, exist_ok=True)
    
    input_basename = os.path.basename(media_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"live_{input_basename}_{int(time.time())}"
    
    temp_dir = os.path.join(args.result_dir, "v15")
    os.makedirs(temp_dir, exist_ok=True)
    
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, input_basename + ".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)
    
    output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
    
    # 1. Prepare visual frames
    progress(0.12, desc="🎞️ สกัดเฟรมภาพอวตาร...")
    is_video = (get_file_type(media_path) == "video")
    save_dir_full = None
    if is_video:
        save_dir_full = os.path.join(temp_dir, f"frames_{input_basename}")
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f'ffmpeg -y -v fatal -i "{media_path}" -start_number 0 "{save_dir_full}/%08d.png"'
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(media_path)
    else:
        input_img_list = [media_path]
        fps = args.fps
        
    if not input_img_list:
        return None, "❌ ไม่สามารถโหลดเฟรมจากอวตารได้"
    
    # 2. Extract Audio features
    progress(0.22, desc="🔊 วิเคราะห์คลื่นเสียง (Whisper)...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
    
    # 3. Detect Face Landmarks & Bounding Box
    progress(0.35, desc="👤 วิเคราะห์โครงสร้างใบหน้าและริมฝีปาก...")
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        with open(crop_coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
            
    if not frame_list:
        return None, "❌ ไม่พบใบหน้าที่ชัดเจนในรูปภาพ/วิดีโอ"
    
    # 4. Latent encoding
    progress(0.45, desc="🧮 กำลังเตรียม Latent VAE...")
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + args.extra_margin, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
        
    if not input_latent_list:
        return None, "❌ ตรวจจับกรอบใบหน้าไม่สำเร็จ"
    
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    # 5. UNet Inference
    progress(0.55, desc="🎨 กำลังสร้างการเคลื่อนไหวริมฝีปาก (UNet)...")
    video_num = len(whisper_chunks)
    gen = datagen(whisper_chunks, input_latent_list_cycle, args.batch_size)
    res_frame_list = []
    
    for i, (whisper_batch, latent_batch) in enumerate(gen):
        audio_feature_batch = pe(whisper_batch.to(device))
        latent_batch = latent_batch.to(dtype=weight_dtype, device=device)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
        
        current_prog = 0.55 + (0.25 * min(len(res_frame_list), video_num) / max(video_num, 1))
        progress(min(current_prog, 0.80), desc=f"🎨 สร้างเฟรม {min(len(res_frame_list), video_num)}/{video_num}...")
    
    # 6. Face Blending
    progress(0.82, desc="🖼️ ผสานใบหน้าความละเอียดสูง...")
    for i, res_frame in enumerate(res_frame_list):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + args.extra_margin, ori_frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
        except Exception:
            continue
            
    # 7. Render final MP4 with Audio
    progress(0.92, desc="📹 เรนเดอร์วิดีโอสมบูรณ์...")
    temp_vid_path = os.path.join(temp_dir, f"silent_{output_basename}.mp4")
    cmd_render = f'ffmpeg -y -v warning -r {fps} -f image2 -i "{result_img_save_path}/%08d.png" -vcodec libx264 -vf format=yuv420p -crf 18 "{temp_vid_path}"'
    os.system(cmd_render)
    
    cmd_audio = f'ffmpeg -y -v warning -i "{temp_vid_path}" -i "{audio_path}" -c:v copy -c:a aac -shortest "{output_vid_name}"'
    os.system(cmd_audio)
    
    # Cleanup temp frames
    shutil.rmtree(result_img_save_path, ignore_errors=True)
    if os.path.exists(temp_vid_path):
        os.remove(temp_vid_path)
    if save_dir_full and os.path.exists(save_dir_full):
        shutil.rmtree(save_dir_full, ignore_errors=True)
        
    if os.path.exists(output_vid_name):
        progress(1.0, desc="✅ สร้างวิดีโอสำเร็จ!")
        return output_vid_name, "🎉 สร้างวิดีโอ Lip-Sync สำเร็จ พร้อมไลฟ์สดแล้ว!"
    else:
        return None, "❌ เกิดข้อผิดพลาดในการรวมไฟล์วิดีโอ"

# ═══════════════════════════════════════════════════════════════
# Unified Workflow
# ═══════════════════════════════════════════════════════════════
def run_live_pipeline(avatar_file, script_text, voice_style, clone_audio, progress=gr.Progress()):
    """Run full workflow: Script validation → TTS Audio → Lip-Sync Video"""
    if not avatar_file:
        return None, None, "❌ กรุณาเลือกอวตารผู้จัดไลฟ์ (คลิกปุ่ม '✨ ใช้รูป ai.png' ด้านบนได้เลย)"
    if not script_text or not script_text.strip():
        return None, None, "❌ กรุณาสร้างสคริปต์ หรือพิมพ์ข้อความที่ต้องการพูดก่อน"
        
    # Clean text
    clean_text = script_text.strip()
    clean_text = re.sub(r'[*_#>`\[\]()~]', '', clean_text).strip()
    # Pick first paragraph or max 250 chars for optimal performance
    paragraphs = [p.strip() for p in clean_text.split('\n') if p.strip()]
    speech_segment = paragraphs[0] if paragraphs else clean_text
    if len(speech_segment) > 280:
        speech_segment = speech_segment[:280] + "..."
        
    # 1. TTS
    progress(0.1, desc="[1/2] กำลังพากย์เสียงภาษาไทย...")
    audio_path, tts_msg = generate_tts(speech_segment, voice_style, clone_audio, progress)
    if not audio_path:
        return None, None, f"❌ ไม่สามารถสร้างเสียงได้: {tts_msg}"
        
    # 2. Lip-Sync
    progress(0.4, desc="[2/2] กำลัง Lip-Sync วิดีโอ 9:16...")
    video_path, lip_msg = lipsync_inference(avatar_file, audio_path, progress)
    if not video_path:
        return None, audio_path, f"❌ สร้างวิดีโอไม่สำเร็จ: {lip_msg}"
        
    status_summary = f"🔴 สตรีมสดสำเร็จ! • ข้อความ: \"{speech_segment[:60]}...\""
    return video_path, audio_path, status_summary

# ═══════════════════════════════════════════════════════════════
# High-End Studio UI Styling (Modern TikTok Live Studio)
# ═══════════════════════════════════════════════════════════════
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Thai:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0d14;
    --bg-card: rgba(18, 24, 38, 0.85);
    --bg-card-hover: rgba(24, 32, 50, 0.95);
    --border-color: rgba(255, 255, 255, 0.08);
    --border-highlight: rgba(0, 242, 254, 0.3);
    --tiktok-pink: #fe2c55;
    --tiktok-cyan: #00f2fe;
    --tiktok-gradient: linear-gradient(135deg, #fe2c55 0%, #ff0055 50%, #d62246 100%);
    --cyan-gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    --gold-accent: #ffb800;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

* {
    font-family: 'Plus Jakarta Sans', 'Noto Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(254, 44, 85, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(0, 242, 254, 0.06) 0%, transparent 40%),
        linear-gradient(180deg, #080a10 0%, #0d111a 100%) !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* ── Top Header Bar ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 18px;
    background: rgba(15, 20, 32, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    margin-bottom: 12px;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo-badge {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: var(--tiktok-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 14px rgba(254, 44, 85, 0.4);
}

.brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.3px;
    margin: 0;
    background: linear-gradient(90deg, #ffffff 40%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin: 0;
}

.header-badges {
    display: flex;
    align-items: center;
    gap: 8px;
}

.sys-badge {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-color);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.badge-live {
    background: rgba(254, 44, 85, 0.15);
    border: 1px solid rgba(254, 44, 85, 0.4);
    color: #ff4b72;
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 700;
}

.live-indicator-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #fe2c55;
    animation: live-blink 1.4s ease-in-out infinite;
}

@keyframes live-blink {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 8px #fe2c55; }
    50% { opacity: 0.4; transform: scale(0.85); box-shadow: none; }
}

/* ── Panel Cards ── */
.control-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 14px !important;
    padding: 14px !important;
    backdrop-filter: blur(16px);
    margin-bottom: 10px;
    transition: border-color 0.2s ease;
}

.control-card:hover {
    border-color: rgba(255, 255, 255, 0.14) !important;
}

.card-title-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    font-weight: 700;
    color: #f1f5f9;
}

.step-num {
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: rgba(0, 242, 254, 0.15);
    color: var(--tiktok-cyan);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 800;
    border: 1px solid rgba(0, 242, 254, 0.3);
}

/* ── Clean Inputs ── */
textarea, input[type="text"], .gr-text-input {
    background: rgba(12, 16, 26, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    font-size: 0.84rem !important;
    padding: 8px 12px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--tiktok-cyan) !important;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.2) !important;
    outline: none !important;
}

.gr-dropdown {
    background: rgba(12, 16, 26, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
}

label, .gr-input-label {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}

/* ── Quick Chips Buttons ── */
.chip-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 20px !important;
    color: #cbd5e1 !important;
    font-size: 0.74rem !important;
    padding: 4px 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.chip-btn:hover {
    background: rgba(0, 242, 254, 0.15) !important;
    border-color: rgba(0, 242, 254, 0.4) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

.chip-primary {
    background: rgba(254, 44, 85, 0.12) !important;
    border-color: rgba(254, 44, 85, 0.3) !important;
    color: #ff6b87 !important;
}

.chip-primary:hover {
    background: rgba(254, 44, 85, 0.25) !important;
    color: #ffffff !important;
}

/* ── Buttons ── */
.btn-generate {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.btn-generate:hover {
    background: linear-gradient(135deg, #2a384f 0%, #475569 100%) !important;
    border-color: var(--tiktok-cyan) !important;
    transform: translateY(-1px);
}

.btn-main-action {
    background: var(--tiktok-gradient) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 12px 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(254, 44, 85, 0.4) !important;
    letter-spacing: 0.2px !important;
    transition: all 0.25s ease !important;
}

.btn-main-action:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 6px 28px rgba(254, 44, 85, 0.6) !important;
}

/* ── TikTok Phone Stage Mockup ── */
.phone-mockup-wrapper {
    position: relative;
    max-width: 320px;
    margin: 0 auto;
    border-radius: 28px;
    padding: 8px;
    background: linear-gradient(180deg, #1f2738 0%, #0d121c 100%);
    box-shadow: 
        0 12px 40px rgba(0, 0, 0, 0.6),
        0 0 0 1px rgba(255, 255, 255, 0.1),
        0 0 30px rgba(254, 44, 85, 0.12);
}

.phone-screen {
    position: relative;
    border-radius: 22px;
    overflow: hidden;
    background: #000000;
    aspect-ratio: 9/16;
    display: flex;
    flex-direction: column;
}

/* Floating TikTok Live Overlays */
.stage-top-overlay {
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 10;
    pointer-events: none;
}

.stage-host-tag {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(8px);
    padding: 3px 8px 3px 4px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.host-avatar-mini {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--tiktok-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 800;
    color: #fff;
}

.host-handle {
    font-size: 0.68rem;
    font-weight: 700;
    color: #ffffff;
}

.stage-live-tag {
    background: #fe2c55;
    color: #ffffff;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    box-shadow: 0 0 8px rgba(254, 44, 85, 0.8);
}

.stage-viewers {
    background: rgba(0, 0, 0, 0.5);
    color: #e2e8f0;
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 0.65rem;
}

/* Floating Actions on Right */
.stage-side-actions {
    position: absolute;
    right: 8px;
    bottom: 95px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    z-index: 10;
    pointer-events: none;
}

.action-bubble {
    display: flex;
    flex-direction: column;
    align-items: center;
    color: #ffffff;
    font-size: 0.62rem;
    font-weight: 700;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}

.action-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(6px);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    margin-bottom: 2px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Floating Chat Overlay */
.stage-chat-overlay {
    position: absolute;
    left: 10px;
    right: 48px;
    bottom: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    z-index: 10;
    pointer-events: none;
}

.chat-bubble {
    background: rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(6px);
    border-radius: 8px;
    padding: 3px 8px;
    font-size: 0.65rem;
    color: #ffffff;
    max-width: 90%;
    line-height: 1.25;
}

.chat-user {
    font-weight: 700;
    color: var(--tiktok-cyan);
    margin-right: 4px;
}

/* Video Player inside phone */
.phone-mockup-wrapper video {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    border-radius: 20px !important;
}

/* Accordion */
.gr-accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    background: rgba(12, 16, 26, 0.4) !important;
}

/* Tabs */
.gr-tabs {
    background: transparent !important;
}

.gr-tab-nav {
    border-bottom: 1px solid var(--border-color) !important;
}

/* Status Banner */
.status-pill {
    padding: 6px 12px;
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--border-color);
    font-size: 0.78rem;
    font-weight: 500;
    color: #cbd5e1;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }
"""

# ═══════════════════════════════════════════════════════════════
# Build Gradio UI
# ═══════════════════════════════════════════════════════════════
llm_choices = get_llm_choices()
default_model = llm_choices[0] if llm_choices else None

default_system_prompt = """คุณคือนักไลฟ์ขายของออนไลน์มืออาชีพ (TikTok Live Style)
พูดภาษาไทย น้ำเสียงสดใส เป็นกันเอง น่าฟัง กระตุ้นให้ผู้ชมกดสั่งซื้อ
เขียนสคริปต์สั้นกระชับ 3-4 ประโยคต่อช่วง ไม่ใส่อีโมจิที่ AI อ่านไม่ได้ และไม่ใส่ markdown ซับซ้อน"""

with gr.Blocks(title="Live AI Avatar Studio PRO") as demo:
    
    # ── Top Bar ──
    gr.HTML("""
    <div class="app-header">
        <div class="brand-wrapper">
            <div class="brand-logo-badge">⚡</div>
            <div>
                <h1 class="brand-title">LIVE AI AVATAR STUDIO</h1>
                <p class="brand-subtitle">สตูดิโอไลฟ์สดอวตาร AI อัจฉริยะ • สคริปต์ LLM + เสียงพากย์ไทย + ขยับปากสมจริง</p>
            </div>
        </div>
        <div class="header-badges">
            <div class="sys-badge">⚡ MuseTalk 1.5</div>
            <div class="sys-badge">🎙️ OmniVoice TH</div>
            <div class="sys-badge badge-live">
                <span class="live-indicator-dot"></span>
                <span>STUDIO READY</span>
            </div>
        </div>
    </div>
    """)
    
    with gr.Row(equal_height=False):
        
        # ═══════════════════════════════════════════════════════════
        # LEFT COLUMN: Studio Controls (Clear 3 Steps)
        # ═══════════════════════════════════════════════════════════
        with gr.Column(scale=6, min_width=420):
            
            # ── CARD 1: Host Avatar Selection ──
            with gr.Group(elem_classes=["control-card"]):
                with gr.Row():
                    gr.HTML("""
                    <div class="card-title-bar">
                        <span class="step-num">1</span>
                        <span>เลือกผู้จัดไลฟ์ (Host Avatar)</span>
                    </div>
                    """)
                    sample_img_btn = gr.Button("✨ ใช้ภาพตัวอย่าง (ai.png)", size="sm", elem_classes=["chip-btn", "chip-primary"])
                
                with gr.Tabs():
                    with gr.TabItem("📷 รูปภาพอวตาร (แนะนำ)"):
                        avatar_input = gr.Image(
                            label="อัปโหลดรูปภาพคนหรือตัวละคร (หน้าตรง ชัดเจน)",
                            type="filepath",
                            sources=['upload', 'clipboard'],
                            height=170
                        )
                    with gr.TabItem("🎬 วิดีโออวตาร"):
                        avatar_video_input = gr.Video(
                            label="อัปโหลดคลิปวิดีโอสั้นของคนพูด",
                            sources=['upload'],
                            height=170
                        )
            
            # ── CARD 2: AI Script & Product Brain ──
            with gr.Group(elem_classes=["control-card"]):
                gr.HTML("""
                <div class="card-title-bar">
                    <span class="step-num">2</span>
                    <span>สมอง AI & สคริปต์ไลฟ์สด (AI Script)</span>
                </div>
                """)
                
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=llm_choices,
                        value=default_model,
                        label="โมเดลสมอง AI (จาก api.json)",
                        interactive=True,
                        scale=4
                    )
                    test_brain_btn = gr.Button("⚡ ทดสอบโมเดล", size="sm", scale=1, elem_classes=["btn-generate"])
                
                # Compact status badge for test
                brain_test_status = gr.HTML("<div style='font-size:0.75rem; color:#94a3b8; margin-top:-4px; margin-bottom:8px;'>กดปุ่ม 'ทดสอบโมเดล' เพื่อเช็คความพร้อมและความเร็วของ API</div>")
                
                # Quick Product Presets
                gr.HTML("<div style='font-size:0.76rem; font-weight:600; color:#cbd5e1; margin-bottom:4px;'>เลือกตัวอย่างสินค้าด่วน:</div>")
                with gr.Row():
                    btn_preset_skin = gr.Button("🧴 สกินแคร์ / MCT Oil", size="sm", elem_classes=["chip-btn"])
                    btn_preset_fashion = gr.Button("👗 แฟชั่น / เสื้อผ้า", size="sm", elem_classes=["chip-btn"])
                    btn_preset_gadget = gr.Button("📱 ไอที / หูฟังไร้สาย", size="sm", elem_classes=["chip-btn"])
                
                product_info = gr.Textbox(
                    label="ข้อมูลสินค้า / จุดเด่น / ราคาโปรโมชัน",
                    placeholder="พิมพ์ชื่อสินค้า จุดเด่น หรือโปรโมชันที่ต้องการขาย...",
                    lines=2
                )
                
                with gr.Row():
                    gen_script_btn = gr.Button("🪄 คิดสคริปต์ไลฟ์สดด้วย AI", elem_classes=["btn-generate"], scale=2)
                
                script_box = gr.Textbox(
                    label="สคริปต์ที่พูด (ปรับแต่งหรือพิมพ์เองได้อิสระ)",
                    placeholder="สคริปต์สำหรับพูดจะแสดงที่นี่ สามารถแก้ไขข้อความได้ตามใจชอบ...",
                    lines=4,
                    interactive=True
                )
                
                with gr.Accordion("⚙️ ปรับแต่ง Prompt ระบบ (System Prompt)", open=False):
                    system_prompt_box = gr.Textbox(
                        value=default_system_prompt,
                        label="คำสั่งกำกับบุคลิก AI",
                        lines=2
                    )
            
            # ── CARD 3: Voice Studio ──
            with gr.Group(elem_classes=["control-card"]):
                gr.HTML("""
                <div class="card-title-bar">
                    <span class="step-num">3</span>
                    <span>สตูดิโอเสียงพากย์ (Thai Voice Studio)</span>
                </div>
                """)
                
                with gr.Row():
                    voice_choice = gr.Dropdown(
                        choices=VOICE_STYLES,
                        value=VOICE_STYLES[0],
                        label="เลือกสไตล์เสียงพากย์",
                        scale=3,
                        interactive=True
                    )
                    preview_audio_btn = gr.Button("🔊 ทดลองฟังเสียง", size="sm", scale=1, elem_classes=["btn-generate"])
                
                # Clone Audio Uploader (hidden by default)
                clone_audio_input = gr.Audio(
                    label="อัปโหลดไฟล์เสียงต้นแบบที่ต้องการโคลน (WAV / MP3)",
                    type="filepath",
                    visible=False
                )
                
                # TTS Preview Player
                tts_audio_preview = gr.Audio(
                    label="ไฟล์เสียงที่สร้างเสร็จ",
                    type="filepath",
                    interactive=False,
                    visible=True
                )
        
        # ═══════════════════════════════════════════════════════════
        # RIGHT COLUMN: TikTok Live Stage (9:16 Simulator)
        # ═══════════════════════════════════════════════════════════
        with gr.Column(scale=5, min_width=330):
            
            # Phone Bezel Mockup
            with gr.Group(elem_classes=["phone-mockup-wrapper"]):
                
                # Live Top Overlay
                gr.HTML("""
                <div class="stage-top-overlay">
                    <div class="stage-host-tag">
                        <div class="host-avatar-mini">AI</div>
                        <div class="host-handle">@live.ai.shop</div>
                    </div>
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span class="stage-live-tag">LIVE</span>
                        <span class="stage-viewers">👥 2.4k</span>
                    </div>
                </div>
                """)
                
                # Side Actions Overlay
                gr.HTML("""
                <div class="stage-side-actions">
                    <div class="action-bubble">
                        <div class="action-icon">❤️</div>
                        <span>18.5k</span>
                    </div>
                    <div class="action-bubble">
                        <div class="action-icon">💬</div>
                        <span>642</span>
                    </div>
                    <div class="action-bubble">
                        <div class="action-icon">🎁</div>
                        <span>ของขวัญ</span>
                    </div>
                    <div class="action-bubble">
                        <div class="action-icon" style="background:#ffb800; border-color:#ffe082;">🛍️</div>
                        <span style="color:#ffb800;">ตะกร้า #1</span>
                    </div>
                </div>
                """)
                
                # Live Chat Overlay
                gr.HTML("""
                <div class="stage-chat-overlay">
                    <div class="chat-bubble"><span class="chat-user">somchai_99:</span> จัดส่งด่วนไหมครับแอดมิน</div>
                    <div class="chat-bubble"><span class="chat-user">fah_beauty:</span> กดสั่งโปร 1 แถม 1 ในตะกร้าแล้วค่า ✨</div>
                    <div class="chat-bubble"><span class="chat-user">k_nont:</span> ใช้ดีมากครับ กลิ่นหอม ทานง่าย</div>
                </div>
                """)
                
                # 9:16 Video Player
                output_video_player = gr.Video(
                    label="",
                    show_label=False,
                    interactive=False,
                    height=460,
                    elem_classes=["phone-screen"]
                )
            
            # Action Button & Status
            start_live_btn = gr.Button("🔴 เริ่มไลฟ์สด / สร้างวิดีโอ (Start Live)", elem_classes=["btn-main-action"])
            
            live_status_box = gr.HTML("""
            <div class="status-pill" style="margin-top:8px; text-align:center;">
                พร้อมสร้างวิดีโอไลฟ์สด • กรอกข้อมูลแล้วกดปุ่มสีแดงด้านบนได้เลย
            </div>
            """)

    # ═══════════════════════════════════════════════════════════
    # Interactive Event Callbacks
    # ═══════════════════════════════════════════════════════════
    
    # 1. Quick Load Sample Image (ai.png)
    def load_sample_avatar():
        if os.path.exists(SAMPLE_AVATAR_PATH):
            return SAMPLE_AVATAR_PATH
        return None
        
    sample_img_btn.click(
        fn=load_sample_avatar,
        inputs=[],
        outputs=[avatar_input]
    )
    
    # 2. Preset product buttons
    btn_preset_skin.click(lambda: PRESET_PRODUCTS["skincare"], outputs=[product_info])
    btn_preset_fashion.click(lambda: PRESET_PRODUCTS["fashion"], outputs=[product_info])
    btn_preset_gadget.click(lambda: PRESET_PRODUCTS["gadget"], outputs=[product_info])
    
    # 3. AI Brain Test Button
    def handle_test_brain(model):
        badge, sample = test_ai_brain(model)
        color = "#10b981" if "พร้อมใช้งาน" in badge else "#ef4444"
        sample_html = ""
        if sample:
            sample_clean = sample[:90].replace('"', "'")
            sample_html = f"<div style='color:#94a3b8; font-size:0.72rem; margin-top:2px;'>ตัวอย่างตอบ: '{sample_clean}...'</div>"
        html = f"""<div style="font-size:0.78rem; color:{color}; padding:6px 10px; background:rgba(255,255,255,0.04); border-radius:8px; border:1px solid rgba(255,255,255,0.08); margin-top:2px;">
            {badge}
            {sample_html}
        </div>"""
        return html
        
    test_brain_btn.click(
        fn=handle_test_brain,
        inputs=[model_choice],
        outputs=[brain_test_status]
    )
    
    # 4. Generate Script
    gen_script_btn.click(
        fn=generate_script_llm,
        inputs=[model_choice, system_prompt_box, product_info],
        outputs=[script_box]
    )
    
    # 5. Toggle Voice Clone Uploader
    def handle_voice_style_change(style):
        is_clone = VOICE_STYLE_MAP.get(style, "") == "__clone__"
        return gr.update(visible=is_clone)
        
    voice_choice.change(
        fn=handle_voice_style_change,
        inputs=[voice_choice],
        outputs=[clone_audio_input]
    )
    
    # 6. Preview Voice Only
    def handle_voice_preview(text, style, clone_audio):
        if not text or not text.strip():
            return None
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        short_text = paragraphs[0] if paragraphs else text
        if len(short_text) > 120:
            short_text = short_text[:120]
        audio_path, _ = generate_tts(short_text, style, clone_audio)
        return audio_path
        
    preview_audio_btn.click(
        fn=handle_voice_preview,
        inputs=[script_box, voice_choice, clone_audio_input],
        outputs=[tts_audio_preview]
    )
    
    # 7. Unified Avatar Resolver (Image or Video)
    def resolve_and_run(img_path, vid_path, script, voice, clone_audio):
        avatar = img_path if img_path else vid_path
        video_out, audio_out, status = run_live_pipeline(avatar, script, voice, clone_audio)
        
        status_html = f"""<div class="status-pill" style="margin-top:8px; text-align:center; color:#38bdf8;">
            {status}
        </div>"""
        return video_out, audio_out, status_html
        
    start_live_btn.click(
        fn=resolve_and_run,
        inputs=[avatar_input, avatar_video_input, script_box, voice_choice, clone_audio_input],
        outputs=[output_video_player, tts_audio_preview, live_status_box]
    )

# ═══════════════════════════════════════════════════════════════
# Launch Server
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
args = parser.parse_args()

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print("\n[*] Live AI Avatar Studio PRO starting...")
print(f"[*] Open http://{args.ip}:{args.port}")

demo.queue().launch(
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
    share=args.share,
    server_name=args.ip,
    server_port=args.port
)
