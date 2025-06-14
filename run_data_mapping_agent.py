#!/usr/bin/env python3
"""
Run the Advanced Data Mapping Agent
Simple runner script that starts the FastAPI server
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from data_mapping_agent import main

    if __name__ == "__main__":
        # Run the application
        asyncio.run(main())

except KeyboardInterrupt:
    print("\n🛑 Data Mapping Agent stopped by user")
except Exception as e:
    print(f"❌ Error starting Data Mapping Agent: {e}")
    sys.exit(1)