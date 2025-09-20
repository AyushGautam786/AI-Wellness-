import os
import sys

print("🔍 DIAGNOSTIC TEST")
print("=" * 40)

# Check current directory
print(f"Current working directory: {os.getcwd()}")

# Check if files exist
files_to_check = ['app.py', 'best_model.pth', 'requirements.txt']
for file in files_to_check:
    if os.path.exists(file):
        print(f"✓ {file} exists")
    else:
        print(f"❌ {file} NOT found")

# Test imports
print("\n📦 TESTING IMPORTS:")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")

try:
    import gradio as gr
    print(f"✓ Gradio {gr.__version__}")
except ImportError as e:
    print(f"❌ Gradio: {e}")

# Test GPU availability
print(f"\n🖥️  CUDA available: {torch.cuda.is_available() if 'torch' in locals() else 'N/A'}")

print("\n🚀 If all imports are ✓, the app should work!")
print("Run: python app.py")