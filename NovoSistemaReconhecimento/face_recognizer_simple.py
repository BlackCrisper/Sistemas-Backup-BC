"""
Módulo para reconhecimento facial usando OpenCV básico
Versão simplificada que não requer opencv-contrib-python
Usa comparação de características faciais simples
"""
import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict
from database import Database


class FaceRecognizer:
    def __init__(self, database: Database, threshold: float = 0.6):
        """
        Inicializa o reconhecedor facial usando comparação de características
        
        Args:
            database: Instância do Database para buscar dados
            threshold: Threshold de similaridade (0-1, maior = mais rigoroso)
        """
        self.database = database
        self.threshold = threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}  # {usuario_id: lista de características}
        self.load_known_faces()
    
    def load_known_faces(self):
        """Carrega todas as faces conhecidas do banco de dados"""
        self.known_faces = {}
        
        usuarios = self.database.listar_usuarios()
        for usuario in usuarios:
            usuario_id = usuario['id']
            user_dir = os.path.join('faces', str(usuario_id))
            
            if not os.path.exists(user_dir):
                continue
            
            face_features = []
            for filename in os.listdir(user_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(user_dir, filename)
                    features = self.extract_features(image_path)
                    if features is not None:
                        face_features.append(features)
            
            if face_features:
                self.known_faces[usuario_id] = face_features
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extrai características de uma imagem facial
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Array de características ou None
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # Redimensiona para tamanho padrão
        image = cv2.resize(image, (200, 200))
        
        # Aplica equalização de histograma para melhorar contraste
        image = cv2.equalizeHist(image)
        
        # Extrai características usando LBP (Local Binary Pattern) simplificado
        # Divide a imagem em blocos e calcula histograma de cada bloco
        features = []
        block_size = 50
        for y in range(0, 200, block_size):
            for x in range(0, 200, block_size):
                block = image[y:y+block_size, x:x+block_size]
                # Calcula histograma do bloco
                hist = cv2.calcHist([block], [0], None, [32], [0, 256])
                features.extend(hist.flatten())
        
        # Normaliza o vetor de características
        features = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def extract_features_from_array(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai características de um array de imagem
        
        Args:
            face_image: Imagem do rosto em escala de cinza
            
        Returns:
            Array de características ou None
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Redimensiona para tamanho padrão
        gray = cv2.resize(gray, (200, 200))
        
        # Aplica equalização
        gray = cv2.equalizeHist(gray)
        
        # Extrai características
        features = []
        block_size = 50
        for y in range(0, 200, block_size):
            for x in range(0, 200, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                hist = cv2.calcHist([block], [0], None, [32], [0, 256])
                features.extend(hist.flatten())
        
        features = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compara duas características e retorna similaridade (0-1)
        
        Args:
            features1: Primeira característica
            features2: Segunda característica
            
        Returns:
            Similaridade entre 0 e 1 (1 = idêntico)
        """
        # Usa correlação de Pearson (cosine similarity para vetores normalizados)
        similarity = np.dot(features1, features2)
        return float(similarity)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos em uma imagem usando Haar Cascade
        
        Args:
            image: Imagem numpy array (BGR ou grayscale)
            
        Returns:
            Lista de tuplas (x, y, width, height)
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
        Extrai a região do rosto da imagem
        
        Args:
            image: Imagem numpy array (BGR)
            bbox: Tupla (x, y, width, height)
            
        Returns:
            Imagem do rosto ou None
        """
        x, y, w, h = bbox
        
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        
        height, width = image.shape[:2]
        x2 = min(x + w, width)
        y2 = min(y + h, height)
        x = max(0, x)
        y = max(0, y)
        
        face_region = image[y:y2, x:x2]
        return face_region
    
    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Reconhece uma face comparando com faces conhecidas
        
        Args:
            face_image: Imagem do rosto (BGR ou grayscale)
            
        Returns:
            Dicionário com informações do usuário reconhecido ou None
        """
        # Extrai características da face
        features = self.extract_features_from_array(face_image)
        if features is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        # Compara com todas as faces conhecidas
        for usuario_id, known_features_list in self.known_faces.items():
            for known_features in known_features_list:
                similarity = self.compare_features(features, known_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = usuario_id
        
        # Verifica se está acima do threshold
        if best_match and best_similarity >= self.threshold:
            usuario = self.database.buscar_usuario(best_match)
            if usuario:
                return {
                    'usuario_id': best_match,
                    'nome': usuario['nome'],
                    'confidence': best_similarity,
                    'distance': 1.0 - best_similarity
                }
        
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
                        'location': (int(x), int(y), int(w), int(h)),
                        'usuario_id': int(recognition['usuario_id']),
                        'nome': str(recognition['nome']),
                        'confidence': float(recognition['confidence'])
                    })
                else:
                    results.append({
                        'location': (int(x), int(y), int(w), int(h)),
                        'usuario_id': None,
                        'nome': 'Desconhecido',
                        'confidence': 0.0
                    })
        
        return results
    
    def add_face(self, usuario_id: int, face_image: np.ndarray):
        """
        Adiciona uma nova face ao sistema
        
        Args:
            usuario_id: ID do usuário
            face_image: Imagem do rosto
        """
        features = self.extract_features_from_array(face_image)
        if features is not None:
            if usuario_id not in self.known_faces:
                self.known_faces[usuario_id] = []
            self.known_faces[usuario_id].append(features)
