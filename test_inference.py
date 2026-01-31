import os
import sys
import torch
from dotenv import load_dotenv

# Setup paths
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

from configs import Config
from infer.modules.vc.modules import VC
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Initialize Config and VC
print("Initializing Config...")
config = Config()
vc = VC(config)

# Parameters
sid = "egirl.pth"
input_audio = "/Users/nevilpatel/Downloads/Listening/1.mp3"
file_index = "logs/E-Girl/added_IVF2182_Flat_nprobe_1_egirl.index"
f0_up_key = 0
f0_method = "rmvpe"
index_rate = 0.75
filter_radius = 3
resample_sr = 0
rms_mix_rate = 0.25
protect = 0.33

print(f"Loading model: {sid}")
# Load model
vc.get_vc(sid)

print(f"Running inference on: {input_audio}")
# Run inference
info, audio_opt = vc.vc_single(
    sid=0,  # sid is ignored in vc_single if net_g is already loaded, but passed as argument. Wait, vc_single signature: sid, input_audio_path, ...
    input_audio_path=input_audio,
    f0_up_key=f0_up_key,
    f0_file=None,
    f0_method=f0_method,
    file_index=file_index,
    file_index2="",
    index_rate=index_rate,
    filter_radius=filter_radius,
    resample_sr=resample_sr,
    rms_mix_rate=rms_mix_rate,
    protect=protect,
)

print("Inference Result Info:")
print(info)

if "Success" in info:
    print("Inference Successful!")
else:
    print(info)
    import traceback

    traceback.print_exc()
    exit(1)
