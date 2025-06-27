"""
🌿 Plant Disease Detection System Launcher
==========================================

This script launches the Streamlit application and automatically opens it in your default browser.
It provides a convenient way to start the Plant Disease Detection System without manual commands.

Usage:
    python launch.py

The script will:
1. Check if Streamlit is available
2. Launch the Streamlit app on localhost:8501
3. Automatically open the browser to the app
4. Handle errors gracefully with helpful messages

Requirements:
    - streamlit
    - All dependencies listed in requirements.txt
"""

import subprocess
import sys
import os
import time
import webbrowser
import threading
from pathlib import Path

def check_python():
    """Check if Python is available"""
    try:
        version = sys.version.split()[0]
        print(f"✅ Python {version} detected")
        return True
    except:
        print("❌ Python not found")
        return False

def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} found")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        print("📦 Installing requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False
    else:
        print("⚠️ requirements.txt not found")
        return False

def open_browser_delayed():
    """Open browser after a delay"""
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    print("🌐 Browser opened automatically")

def main():
    """Main launcher function"""
    print("🚀 Plant Disease Detection System - Launcher")
    print("=" * 50)
    
    # Check Python
    if not check_python():
        input("❌ Please install Python 3.8+ and try again. Press Enter to exit...")
        return
    
    # Check Streamlit
    if not check_streamlit():
        print("📦 Installing Streamlit...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "streamlit"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ Streamlit installed")
        except:
            print("❌ Failed to install Streamlit")
            if input("Try installing requirements? (y/n): ").lower() == 'y':
                if not install_requirements():
                    input("❌ Installation failed. Press Enter to exit...")
                    return
    
    # Check if app file exists
    app_file = Path("streamlit_app.py")
    if not app_file.exists():
        print(f"❌ {app_file} not found in current directory")
        print(f"📁 Current directory: {os.getcwd()}")
        input("Press Enter to exit...")
        return
    
    print("✅ All checks passed")
    print("🚀 Starting Streamlit application...")
    print("🌐 App will be available at: http://localhost:8501")
    print("🔍 Browser will open automatically...")
    print("💡 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start browser in background
    browser_thread = threading.Thread(target=open_browser_delayed)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
