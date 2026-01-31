#!/usr/bin/env python3
"""
Simplified RVC WebUI startup script for macOS
This bypasses some fairseq-related components temporarily
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()
load_dotenv("sha256.env")

# macOS optimizations
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

# Set up temp directories
import shutil

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "weights"), exist_ok=True)
os.environ["TEMP"] = tmp

print("RVC WebUI starting...")
print("Environment configured for macOS")

# Import required modules
try:
    import torch

    print(f"PyTorch version: {torch.__version__}")

    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Metal) acceleration available")
    else:
        print("⚠ MPS not available, using CPU")

    import gradio as gr

    print(f"✓ Gradio version: {gr.__version__}")

    # Try to create a simple interface
    def hello(name):
        return f"Hello {name}! RVC WebUI is working on macOS."

    # Create interface
    iface = gr.Interface(
        fn=hello,
        inputs=gr.Textbox(label="Name"),
        outputs="text",
        title="RVC WebUI - macOS Test",
        description="This confirms that the basic setup is working. The full RVC interface will be available once all dependencies are resolved.",
    )

    print("\n" + "=" * 50)
    print("🎉 SUCCESS! RVC WebUI is starting...")
    print("📱 Open your browser to: http://localhost:7865")
    print("=" * 50 + "\n")

    # Launch the interface
    iface.launch(server_name="0.0.0.0", server_port=7865, share=False, inbrowser=True)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some dependencies may still be missing.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting RVC WebUI: {e}")
    sys.exit(1)
