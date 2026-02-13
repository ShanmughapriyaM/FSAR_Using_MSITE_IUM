import importlib
import sys

required_packages = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",          # opencv-python
    "sklearn",
    "matplotlib",
    "scipy","mediapipe",
    "tqdm"
]

print("🔍 Checking required packages...\n")

missing = []

for pkg in required_packages:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} : OK")
    except ImportError:
        print(f"❌ {pkg} : NOT FOUND")
        missing.append(pkg)

print("\n----------------------------------")

if not missing:
    print("🎉 All required packages are installed and ready!")
else:
    print("⚠️ Missing packages:")
    for m in missing:
        print(f"   - {m}")
    print("\nInstall missing packages before proceeding.")

print("----------------------------------")

print("\n🐍 Python version:", sys.version)
