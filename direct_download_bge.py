#!/usr/bin/env python3
"""
Direct BGE Large v1.5 Download (Skip Connectivity Test)
Since connectivity test passed, let's download directly
"""

import os
import sys
from pathlib import Path


def download_bge_model():
    """Download BGE Large v1.5 model directly"""
    print("🎯 Direct BGE Large v1.5 Download")
    print("=" * 50)
    print("📥 Downloading BGE Large v1.5...")
    print("📊 Model size: ~1.34GB")
    print("⏳ This may take 5-15 minutes depending on your connection...")
    print()

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        # Create cache directory
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"📁 Cache directory: {cache_dir}")

        # Enable progress bars and set longer timeouts
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'FALSE'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = 'TRUE'

        # Download and load the model with longer timeout
        print("🔄 Starting BGE Large v1.5 download...")
        print("💡 If download seems stuck, it's likely still downloading in background")

        model = SentenceTransformer(
            'BAAI/bge-large-en-v1.5',
            cache_folder=cache_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )

        # Test the model
        print("\n🧪 Testing model functionality...")
        test_sentences = [
            "This is a test sentence",
            "Schema field mapping example",
            "Customer ID field"
        ]

        embeddings = model.encode(test_sentences)
        print(f"✅ Model test successful!")
        print(f"📊 Embedding shape: {embeddings.shape}")
        print(f"📏 Embedding dimension: {embeddings.shape[1]}")

        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"🎯 BGE Large v1.5 ready on {device}")

        # Calculate similarity test
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"🔗 Similarity test: {similarity:.3f}")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check available disk space (need ~2GB free)")
        print("   2. Try: pip install --upgrade sentence-transformers huggingface_hub")
        print("   3. Check if antivirus is blocking the download")
        print("   4. Try running as administrator")
        print("   5. Clear existing cache: rm -rf model_cache/")
        return False


def main():
    """Main function"""
    try:
        success = download_bge_model()

        print("\n" + "=" * 50)

        if success:
            print("🎉 BGE Large v1.5 downloaded and tested successfully!")
            print("🚀 You can now run: python run_data_mapping_agent.py")
            print("⚡ Future runs will be instant (model is cached)")
        else:
            print("💥 Download failed. Please check the error messages above.")

        return success

    except KeyboardInterrupt:
        print("\n🛑 Download cancelled by user")
        print("💡 You can resume the download by running this script again")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)