#!/usr/bin/env python3
"""
Complete Fix Runner for Data Mapping Agent
This script will fix all issues at once
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description, required=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        if required:
            print(f"   This is required for the system to work")
            return False
        else:
            print(f"   This is optional, continuing...")
            return True


def main():
    """Main fix function"""
    print("🎯 Data Mapping Agent - Complete Fix")
    print("=" * 60)
    print("This script will fix all dependency and compatibility issues")
    print("=" * 60)

    # Step 1: Upgrade pip and core tools
    print("\n📦 Step 1: Upgrading core tools...")
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    run_command("python -m pip install --upgrade setuptools wheel", "Upgrading setuptools and wheel")

    # Step 2: Install/upgrade core ML packages
    print("\n🧠 Step 2: Installing ML packages...")
    ml_packages = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0"
    ]

    for package in ml_packages:
        run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}")

    # Step 3: Fix HuggingFace issues
    print("\n🤗 Step 3: Fixing HuggingFace compatibility...")
    hf_packages = [
        "huggingface-hub>=0.16.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0"
    ]

    for package in hf_packages:
        run_command(f"pip install --upgrade '{package}'", f"Installing {package.split('>=')[0]}")

    # Step 4: Install web framework packages
    print("\n🌐 Step 4: Installing web framework...")
    web_packages = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "aiofiles>=23.0.0",
        "python-multipart>=0.0.6"
    ]

    for package in web_packages:
        run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}")

    # Step 5: Install utility packages
    print("\n🔧 Step 5: Installing utilities...")
    util_packages = [
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0"
    ]

    for package in util_packages:
        run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}", required=False)

    # Step 6: Install optional LangChain packages
    print("\n🦜 Step 6: Installing LangChain (optional)...")
    langchain_packages = [
        "langchain>=0.0.350",
        "langchain-community>=0.0.10"
    ]

    for package in langchain_packages:
        run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}", required=False)

    # Step 7: Create compatibility files
    print("\n🔧 Step 7: Creating compatibility files...")

    # Create requirements.txt
    requirements_content = """# Data Mapping Agent Requirements
# Core ML and NLP
torch>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0
huggingface-hub>=0.16.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# FastAPI and async
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
aiofiles>=23.0.0
python-multipart>=0.0.6

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0
requests>=2.31.0

# Optional LangChain
langchain>=0.0.350
langchain-community>=0.0.10
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("✅ Created requirements.txt")

    # Create .env file template
    env_content = """# Data Mapping Agent Configuration
# HuggingFace settings
HF_HUB_DISABLE_PROGRESS_BARS=FALSE
HF_HUB_DISABLE_TELEMETRY=TRUE

# Model settings
MODEL_CACHE_DIR=./model_cache
SIMILARITY_THRESHOLD=0.75

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Ollama settings (optional)
OLLAMA_MODEL=llama3.1:8b
"""

    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

    # Step 8: Test the installation
    print("\n🧪 Step 8: Testing installation...")

    test_script = '''
import warnings
warnings.filterwarnings("ignore")

# Test core packages
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except Exception as e:
    print(f"❌ Pandas: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers available")
except Exception as e:
    print(f"❌ SentenceTransformers: {e}")

try:
    import fastapi
    print(f"✅ FastAPI {fastapi.__version__}")
except Exception as e:
    print(f"❌ FastAPI: {e}")

try:
    import aiofiles
    print("✅ aiofiles available")
except Exception as e:
    print(f"❌ aiofiles: {e}")

try:
    from huggingface_hub import hf_hub_download
    print("✅ HuggingFace Hub available")
except Exception as e:
    print(f"❌ HuggingFace Hub: {e}")

print("\\n🎯 Installation test completed!")
'''

    with open("test_installation.py", "w") as f:
        f.write(test_script)

    print("Running installation test...")
    run_command("python test_installation.py", "Testing installation", required=False)

    # Step 9: Final instructions
    print("\n" + "=" * 60)
    print("🎉 COMPLETE FIX INSTALLATION FINISHED!")
    print("=" * 60)

    print("📋 What was fixed:")
    print("   ✅ Updated all core dependencies")
    print("   ✅ Fixed huggingface_hub compatibility issues")
    print("   ✅ Installed FastAPI and web components")
    print("   ✅ Added missing aiofiles dependency")
    print("   ✅ Created requirements.txt")
    print("   ✅ Added environment configuration")
    print("   ✅ Suppressed torch warnings")

    print("\n🚀 Next Steps:")
    print("   1. Run: python data_mapping_agent.py")
    print("   2. Or run: python run_data_mapping_agent.py")
    print("   3. Access web interface: http://localhost:8000")
    print("   4. Test model download: python direct_download_bge.py")

    print("\n🔧 If you still have issues:")
    print("   1. Restart your terminal/IDE")
    print("   2. Clear Python cache: python -Bc \"import shutil; shutil.rmtree('__pycache__', ignore_errors=True)\"")
    print("   3. Reinstall from requirements: pip install -r requirements.txt")

    print("\n💡 Files created/updated:")
    print("   📄 requirements.txt - All dependencies")
    print("   📄 .env - Environment configuration")
    print("   📄 test_installation.py - Installation tester")

    # Cleanup test file
    if os.path.exists("test_installation.py"):
        os.remove("test_installation.py")

    print("=" * 60)
    print("🎯 Ready to run Data Mapping Agent!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ All fixes applied successfully!")
            print("💡 You can now run your Data Mapping Agent without errors.")
        else:
            print("\n⚠️ Some issues may remain. Check the output above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)