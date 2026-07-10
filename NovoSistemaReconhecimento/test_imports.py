"""Script de teste para verificar se todas as importações funcionam"""
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    from flask import Flask
    print("✓ Flask")
except ImportError as e:
    print(f"✗ Flask: {e}")

try:
    from database import Database
    print("✓ Database")
except ImportError as e:
    print(f"✗ Database: {e}")

try:
    from face_detector import FaceDetector
    print("✓ FaceDetector")
except ImportError as e:
    print(f"✗ FaceDetector: {e}")

try:
    from face_recognizer_simple import FaceRecognizer
    print("✓ FaceRecognizer")
except ImportError as e:
    print(f"✗ FaceRecognizer: {e}")

print("\nTeste concluído!")
