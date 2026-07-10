"""
Módulo para reconhecimento facial usando OpenCV LBPH
Versão simplificada que não requer dlib ou compilação
"""
import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict
from database import Database


class FaceRecognizer:
    def __init__(self, database: Database, threshold: float = 70.0):
        """
        Inicializa o reconhecedor facial usando OpenCV LBPH
        
        Args:
            database: Instância do Database para buscar dados
            threshold: Threshold de confiança (menor = mais rigoroso, padrão 70.0)
        """
        self.database = database
        self.threshold = threshold
        self.recognizer = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.labels_map = {}  # Mapeia IDs do modelo para usuario_id
        self.load_model()
    
    def load_model(self):
        """Carrega ou cria o modelo LBPH"""
        model_path = 'models/face_recognizer_lbph.yml'
        
        # Cria diretório se não existir
        os.makedirs('models', exist_ok=True)
        
        if os.path.exists(model_path):
            # Carrega modelo existente
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(model_path)
            
            # Carrega mapeamento de labels
            labels_path = 'models/labels_map.pkl'
            if os.path.exists(labels_path):
                with open(labels_path, 'rb') as f:
                    self.labels_map = pickle.load(f)
        else:
            # Cria novo modelo
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.train_model()
    
    def train_model(self):
        """Treina o modelo com todas as faces cadastradas"""
        faces = []
        labels = []
        self.labels_map = {}
        
        # Busca todos os usuários
        usuarios = self.database.listar_usuarios()
        
        if not usuarios:
            return
        
        label_id = 0
        for usuario in usuarios:
            usuario_id = usuario['id']
            user_dir = os.path.join('faces', str(usuario_id))
            
            if not os.path.exists(user_dir):
                continue
            
            # Processa todas as imagens do usuário
            for filename in os.listdir(user_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(user_dir, filename)
                    face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if face_image is not None:
                        # Redimensiona para tamanho padrão
                        face_image = cv2.resize(face_image, (200, 200))
                        faces.append(face_image)
                        labels.append(label_id)
            
            if label_id not in self.labels_map.values():
                self.labels_map[label_id] = usuario_id
                label_id += 1
        
        if faces:
            # Treina o modelo
            self.recognizer.train(faces, np.array(labels))
            
            # Salva modelo e mapeamento
            os.makedirs('models', exist_ok=True)
            self.recognizer.write('models/face_recognizer_lbph.yml')
            
            with open('models/labels_map.pkl', 'wb') as f:
                pickle.dump(self.labels_map, f)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos em uma imagem usando Haar Cascade
        
        Args:
            image: Imagem numpy array (BGR ou grayscale)
            
        Returns:
            Lista de tuplas (x, y, width, height) representando bounding boxes
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extrai e prepara a região do rosto para reconhecimento
        
        Args:
            image: Imagem numpy array (BGR)
            bbox: Tupla (x, y, width, height)
            
        Returns:
            Imagem do rosto em escala de cinza e redimensionada ou None
        """
        x, y, w, h = bbox
        
        # Verifica limites
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        
        height, width = image.shape[:2]
        x2 = min(x + w, width)
        y2 = min(y + h, height)
        x = max(0, x)
        y = max(0, y)
        
        face_region = image[y:y2, x:x2]
        
        # Converte para escala de cinza
        if len(face_region.shape) == 3:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_region
        
        # Redimensiona para tamanho padrão
        face_gray = cv2.resize(face_gray, (200, 200))
        
        return face_gray
    
    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Reconhece uma face usando o modelo LBPH
        
        Args:
            face_image: Imagem do rosto em escala de cinza (200x200)
            
        Returns:
            Dicionário com informações do usuário reconhecido ou None
            {'usuario_id': int, 'nome': str, 'confidence': float}
        """
        if self.recognizer is None or not self.labels_map:
            return None
        
        try:
            # Prediz a face
            label_id, confidence = self.recognizer.predict(face_image)
            
            # LBPH retorna menor confiança = melhor match
            # Converte para porcentagem de confiança (inverte)
            confidence_percent = max(0, 100 - confidence)
            
            # Verifica se está dentro do threshold
            if confidence_percent >= (100 - self.threshold):
                # Busca usuario_id do label
                usuario_id = self.labels_map.get(label_id)
                
                if usuario_id:
                    usuario = self.database.buscar_usuario(usuario_id)
                    if usuario:
                        return {
                            'usuario_id': usuario_id,
                            'nome': usuario['nome'],
                            'confidence': confidence_percent / 100.0,
                            'distance': confidence
                        }
            
            return None
        except Exception as e:
            print(f"Erro ao reconhecer face: {e}")
            return None
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta e reconhece todas as faces em uma imagem
        
        Args:
            image: Imagem numpy array (BGR)
            
        Returns:
            Lista de dicionários com informações das faces reconhecidas
        """
        # Detecta rostos
        face_locations = self.detect_faces(image)
        
        results = []
        for (x, y, w, h) in face_locations:
            # Extrai região do rosto
            face_image = self.extract_face_region(image, (x, y, w, h))
            
            if face_image is not None:
                # Reconhece
                recognition = self.recognize_face(face_image)
                
                if recognition:
                    results.append({
                        'location': (x, y, w, h),
                        'usuario_id': recognition['usuario_id'],
                        'nome': recognition['nome'],
                        'confidence': recognition['confidence']
                    })
                else:
                    results.append({
                        'location': (x, y, w, h),
                        'usuario_id': None,
                        'nome': 'Desconhecido',
                        'confidence': 0.0
                    })
        
        return results
