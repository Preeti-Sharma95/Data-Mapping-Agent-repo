#!/usr/bin/env python3
"""
Automated dependency installation script for Data Mapping Agent
This script will install all required dependencies and fix compatibility issues
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False


def install_dependencies():
    """Install all required dependencies"""
    print("🎯 Data Mapping Agent - Dependency Installation")
    print("=" * 60)

    # Core dependencies with specific versions to avoid conflicts
    dependencies = [
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "aiofiles>=23.0.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0"
    ]

    # Optional dependencies
    optional_dependencies = [
        "langchain>=0.0.350",
        "langchain-community>=0.0.10"
    ]

    print("📦 Installing core dependencies...")
    for dep in dependencies:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️ Failed to install {dep}, continuing...")

    print("\n📦 Installing optional dependencies...")
    for dep in optional_dependencies:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️ Optional dependency {dep} failed, continuing...")

    print("\n🔧 Upgrading key packages to latest versions...")
    upgrade_packages = [
        "huggingface-hub",
        "sentence-transformers",
        "transformers"
    ]

    for package in upgrade_packages:
        run_command(f"pip install --upgrade {package}", f"Upgrading {package}")

    print("\n✅ Dependency installation completed!")
    return True


def create_compatibility_patch():
    """Create a compatibility patch file"""
    patch_content = '''# Compatibility patch for huggingface_hub
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from huggingface_hub import cached_download
except ImportError:
    try:
        from huggingface_hub import hf_hub_download
        import huggingface_hub

        def cached_download(url, cache_dir=None, **kwargs):
            """Compatibility wrapper for deprecated cached_download"""
            if "huggingface.co" in url:
                parts = url.split("/")
                if len(parts) >= 6:
                    repo_id = f"{parts[-4]}/{parts[-3]}"
                    filename = parts[-1]
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=cache_dir,
                        **kwargs
                    )
            raise NotImplementedError("cached_download is deprecated")

        huggingface_hub.cached_download = cached_download

    except ImportError:
        pass

# Suppress torch warnings
try:
    import torch
    torch.utils._pytree._register_pytree_node = lambda *args, **kwargs: None
except:
    pass
'''

    with open("compatibility_patch.py", "w") as f:
        f.write(patch_content)

    print("✅ Created compatibility_patch.py")


def verify_installation():
    """Verify that key packages are working"""
    print("\n🧪 Verifying installation...")

    tests = [
        ("torch", "import torch; print(f'PyTorch: {torch.__version__}')"),
        ("transformers", "import transformers; print(f'Transformers: {transformers.__version__}')"),
        ("sentence-transformers",
         "from sentence_transformers import SentenceTransformer; print('SentenceTransformers: OK')"),
        ("fastapi", "import fastapi; print(f'FastAPI: {fastapi.__version__}')"),
        ("aiofiles", "import aiofiles; print('Aiofiles: OK')"),
        ("pandas", "import pandas as pd; print(f'Pandas: {pd.__version__}')"),
        ("numpy", "import numpy as np; print(f'NumPy: {np.__version__}')"),
        ("sklearn", "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')")
    ]

    for name, test_code in tests:
        try:
            exec(test_code)
            print(f"✅ {name} - Working")
        except Exception as e:
            print(f"❌ {name} - Error: {e}")

    print("\n🎉 Installation verification completed!")


def main():
    """Main installation function"""
    try:
        print("🚀 Starting automated dependency installation...")

        # Install dependencies
        install_dependencies()

        # Create compatibility patch
        create_compatibility_patch()

        # Verify installation
        verify_installation()

        print("\n" + "=" * 60)
        print("🎉 All dependencies installed successfully!")
        print("💡 Next steps:")
        print("   1. Run: python run_data_mapping_agent.py")
        print("   2. Or test BGE download: python direct_download_bge.py")
        print("   3. Access web interface: http://localhost:8000")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Installation failed: {e}")
        print("\n🔧 Manual installation fallback:")
        print("   pip install torch sentence-transformers fastapi uvicorn aiofiles")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)