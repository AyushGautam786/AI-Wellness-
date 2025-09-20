#!/usr/bin/env python3
"""
Simple launcher script that ensures correct working directory
"""
import os
import sys
import subprocess

def main():
    print("🚀 EMOTION CLASSIFIER LAUNCHER")
    print("=" * 40)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = script_dir
    
    print(f"Script directory: {script_dir}")
    
    # Change to the models directory
    os.chdir(models_dir)
    print(f"Changed to: {os.getcwd()}")
    
    # Check if required files exist
    required_files = ['app.py', 'best_model.pth']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"❌ {file} NOT found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        if 'best_model.pth' in missing_files:
            print("💡 Please train the model first by running: train_model.bat")
        return False
    
    print("\n🌐 Starting web application...")
    
    # Run the app
    python_exe = sys.executable
    try:
        result = subprocess.run([python_exe, 'app.py'], cwd=models_dir)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")