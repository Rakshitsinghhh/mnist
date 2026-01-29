import os
import sys
import subprocess

VENV_DIR = "venv"

def run(cmd):
    subprocess.check_call(cmd, shell=True)

print("\n🚀 MNIST Neural Network Setup\n")

# -------------------------------
# Project root (important!)
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# -------------------------------
# Create virtual environment
# -------------------------------
if not os.path.exists(VENV_DIR):
    print("📦 Creating virtual environment...")
    run(f"{sys.executable} -m venv {VENV_DIR}")
else:
    print("✅ Virtual environment already exists")

# -------------------------------
# Get venv python path
# -------------------------------
if os.name == "nt":
    venv_python = os.path.join(VENV_DIR, "Scripts", "python")
else:
    venv_python = os.path.join(VENV_DIR, "bin", "python")

# -------------------------------
# Install dependencies
# -------------------------------
print("\n⬇ Installing required libraries...\n")
run(f"{venv_python} -m pip install --upgrade pip")
run(
    f"{venv_python} -m pip install "
    "numpy matplotlib torch torchvision"
)

# -------------------------------
# User choice
# -------------------------------
print("\nWhat do you want to do?")
print("1️⃣ Train the model (train.py)")
print("2️⃣ Manual testing (manual.py)")

choice = input("\nEnter choice (1 or 2): ").strip()

if choice == "1":
    print("\n🧠 Starting training...\n")
    run(f"{venv_python} train.py")

elif choice == "2":
    print("\n🧪 Running manual testing...\n")
    run(f"{venv_python} manual.py")

else:
    print("\n❌ Invalid choice. Exiting.")

print("\n✅ Process finished.")

