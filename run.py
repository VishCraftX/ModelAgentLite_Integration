#!/usr/bin/env python3
"""
Quick launcher for Multi-Agent ML Integration System
Simple entry point with minimal configuration
"""

import os
import sys

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    print("🚀 Multi-Agent ML Integration System - Quick Start")
    print("=" * 55)
    
    # Check if this is first run
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'demo':
            print("🎬 Running demo mode...")
            os.system("python start_pipeline.py --mode demo")
        elif mode == 'slack':
            print("🤖 Starting Slack bot...")
            os.system("python start_pipeline.py --mode slack")
        elif mode == 'api':
            print("🧪 Starting API testing...")
            os.system("python start_pipeline.py --mode api")
        elif mode == 'test':
            print("🔧 Running tests...")
            os.system("python start_pipeline.py --mode test")
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: demo, slack, api, test")
    else:
        print("Quick start options:")
        print("  python run.py demo   - Run automated demo")
        print("  python run.py slack  - Start Slack bot")
        print("  python run.py api    - Interactive API testing")
        print("  python run.py test   - Run direct tests")
        print("\nOr run the full launcher:")
        print("  python start_pipeline.py")
        
        choice = input("\nPress Enter to run demo, or type 'full' for full launcher: ").strip()
        
        if choice.lower() == 'full':
            os.system("python start_pipeline.py")
        else:
            print("\n🎬 Running automated demo...")
            os.system("python start_pipeline.py --mode demo")

if __name__ == "__main__":
    main()
