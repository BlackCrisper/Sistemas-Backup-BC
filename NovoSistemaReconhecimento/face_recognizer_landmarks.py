"""
Módulo para reconhecimento facial usando landmarks faciais (malha do rosto)
Versão melhorada que extrai características baseadas em geometria facial
"""
import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict
from database import Database


class FaceRecognizer:
    def __init__(self, database: Database, threshold: float = 0.75):
        """
        Inicializa o reconhecedor facial usando landmarks faciais
        
        Args:
            database: Instância do Database para buscar dados
            threshold: Threshold de similaridade (0-1, maior = mais rigoroso)
        """
        self.database = database
        self.threshold = threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
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
    
    def detect_landmarks(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Detecta landmarks faciais (olhos, nariz, boca) e retorna coordenadas
        
        Args:
            face_image: Imagem completa (BGR)
            face_bbox: Bounding box do rosto (x, y, w, h)
            
        Returns:
            Dicionário com landmarks ou None
        """
        x, y, w, h = face_bbox
        
        # Extrai região do rosto
        face_region = face_image[y:y+h, x:x+w]
        if face_region.size == 0:
            return None
        
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        landmarks = {}
        
        # Detecta olhos na região superior do rosto
        eyes_region = gray_face[0:int(h*0.6), :]
        eyes = self.eye_cascade.detectMultiScale(eyes_region, 1.1, 3)
        
        if len(eyes) >= 2:
            # Ordena olhos da esquerda para direita
            eyes = sorted(eyes, key=lambda e: e[0])
            landmarks['left_eye'] = (x + eyes[0][0] + eyes[0][2]//2, y + eyes[0][1] + eyes[0][3]//2)
            landmarks['right_eye'] = (x + eyes[1][0] + eyes[1][2]//2, y + eyes[1][1] + eyes[1][3]//2)
        else:
            # Se não detectar olhos, estima posições baseadas na geometria do rosto
            landmarks['left_eye'] = (x + int(w*0.3), y + int(h*0.35))
            landmarks['right_eye'] = (x + int(w*0.7), y + int(h*0.35))
        
        # Estima nariz (centro do rosto, um pouco abaixo dos olhos)
        landmarks['nose'] = (x + w//2, y + int(h*0.5))
        
        # Estima boca (centro, na parte inferior)
        landmarks['mouth'] = (x + w//2, y + int(h*0.7))
        
        # Pontos adicionais da face
        landmarks['chin'] = (x + w//2, y + h)
        landmarks['forehead'] = (x + w//2, y)
        landmarks['left_cheek'] = (x + int(w*0.2), y + int(h*0.6))
        landmarks['right_cheek'] = (x + int(w*0.8), y + int(h*0.6))
        
        return landmarks
    
    def extract_landmark_features(self, landmarks: Dict) -> np.ndarray:
        """
        Extrai características geométricas dos landmarks
        
        Args:
            landmarks: Dicionário com coordenadas dos landmarks
            
        Returns:
            Array de características geométricas
        """
        features = []
        
        # Distâncias entre pontos-chave
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose = np.array(landmarks['nose'])
        mouth = np.array(landmarks['mouth'])
        chin = np.array(landmarks['chin'])
        forehead = np.array(landmarks['forehead'])
        
        # Distância entre olhos (normalizador)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        if eye_distance == 0:
            eye_distance = 1.0
        
        # Normaliza todas as distâncias pela distância entre olhos
        features.append(eye_distance / eye_distance)  # Sempre 1.0 (normalizador)
        features.append(np.linalg.norm(nose - left_eye) / eye_distance)
        features.append(np.linalg.norm(nose - right_eye) / eye_distance)
        features.append(np.linalg.norm(mouth - nose) / eye_distance)
        features.append(np.linalg.norm(chin - mouth) / eye_distance)
        features.append(np.linalg.norm(forehead - left_eye) / eye_distance)
        features.append(np.linalg.norm(forehead - right_eye) / eye_distance)
        
        # Ângulos
        # Ângulo entre olhos e nariz
        vec1 = nose - left_eye
        vec2 = nose - right_eye
        angle1 = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1, 1))
        features.append(angle1)
        
        # Ângulo entre nariz e boca
        vec3 = mouth - nose
        vec4 = chin - mouth
        angle2 = np.arccos(np.clip(np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4)), -1, 1))
        features.append(angle2)
        
        # Razões de aspecto
        features.append(abs(left_eye[0] - right_eye[0]) / abs(left_eye[1] - right_eye[1]) if abs(left_eye[1] - right_eye[1]) > 0 else 1.0)
        features.append(abs(nose[1] - mouth[1]) / abs(left_eye[0] - right_eye[0]) if abs(left_eye[0] - right_eye[0]) > 0 else 1.0)
        
        # Coordenadas normalizadas (relativas ao centro do rosto)
        center = (left_eye + right_eye) / 2
        features.append((nose[0] - center[0]) / eye_distance)
        features.append((nose[1] - center[1]) / eye_distance)
        features.append((mouth[0] - center[0]) / eye_distance)
        features.append((mouth[1] - center[1]) / eye_distance)
        
        return np.array(features, dtype=np.float32)
    
    def extract_texture_features(self, face_image: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Extrai características de textura baseadas nos landmarks
        
        Args:
            face_image: Imagem do rosto (BGR)
            landmarks: Dicionário com coordenadas dos landmarks
            
        Returns:
            Array de características de textura
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Redimensiona para tamanho padrão
        gray = cv2.resize(gray, (200, 200))
        
        # Aplica equalização de histograma
        gray = cv2.equalizeHist(gray)
        
        features = []
        
        # Extrai características de regiões específicas baseadas nos landmarks
        # Região dos olhos
        eye_region = gray[50:100, 50:150]
        if eye_region.size > 0:
            hist_eyes = cv2.calcHist([eye_region], [0], None, [32], [0, 256])
            features.extend(hist_eyes.flatten())
        
        # Região do nariz
        nose_region = gray[100:140, 80:120]
        if nose_region.size > 0:
            hist_nose = cv2.calcHist([nose_region], [0], None, [32], [0, 256])
            features.extend(hist_nose.flatten())
        
        # Região da boca
        mouth_region = gray[140:180, 70:130]
        if mouth_region.size > 0:
            hist_mouth = cv2.calcHist([mouth_region], [0], None, [32], [0, 256])
            features.extend(hist_mouth.flatten())
        
        # LBP simplificado (Local Binary Pattern) em blocos
        block_size = 40
        for y in range(0, 200, block_size):
            for x in range(0, 200, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    hist = cv2.calcHist([block], [0], None, [16], [0, 256])
                    features.extend(hist.flatten())
        
        features = np.array(features, dtype=np.float32)
        # Normaliza
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extrai características completas de uma imagem facial
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Array combinado de características (geometria + textura) ou None
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Detecta rosto
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Usa o primeiro rosto
        face_bbox = faces[0]
        
        # Detecta landmarks
        landmarks = self.detect_landmarks(image, face_bbox)
        if landmarks is None:
            return None
        
        # Extrai características geométricas
        geom_features = self.extract_landmark_features(landmarks)
        
        # Extrai características de textura
        face_region = self.extract_face_region(image, face_bbox)
        if face_region is None:
            return None
        
        texture_features = self.extract_texture_features(face_region, landmarks)
        
        # Combina características
        combined_features = np.concatenate([geom_features, texture_features])
        
        return combined_features
    
    def extract_features_from_array(self, face_image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Extrai características de um array de imagem
        
        Args:
            face_image: Imagem do rosto (BGR)
            face_bbox: Opcional, bounding box do rosto
            
        Returns:
            Array de características ou None
        """
        if face_bbox is None:
            # Detecta rosto se não fornecido
            if len(face_image.shape) == 3:
                full_image = face_image
            else:
                return None
            faces = self.detect_faces(full_image)
            if not faces:
                return None
            face_bbox = faces[0]
            image = full_image
        else:
            image = face_image
            # Se face_image já é a região do rosto, precisa do contexto completo
            # Assumimos que face_image é a imagem completa
            if len(face_image.shape) == 3:
                full_image = face_image
            else:
                return None
        
        # Detecta landmarks
        landmarks = self.detect_landmarks(full_image, face_bbox)
        if landmarks is None:
            return None
        
        # Extrai características geométricas
        geom_features = self.extract_landmark_features(landmarks)
        
        # Extrai características de textura
        face_region = self.extract_face_region(full_image, face_bbox)
        if face_region is None:
            return None
        
        texture_features = self.extract_texture_features(face_region, landmarks)
        
        # Combina características
        combined_features = np.concatenate([geom_features, texture_features])
        
        return combined_features
    
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
        """Detecta rostos em uma imagem"""
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
        """Extrai a região do rosto da imagem"""
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
    
    def recognize_face(self, face_image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict]:
        """
        Reconhece uma face comparando com faces conhecidas
        
        Args:
            face_image: Imagem do rosto (BGR)
            face_bbox: Opcional, bounding box do rosto
            
        Returns:
            Dicionário com informações do usuário reconhecido ou None
        """
        # Extrai características da face
        features = self.extract_features_from_array(face_image, face_bbox)
        if features is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        second_best_similarity = 0.0
        
        # Compara com todas as faces conhecidas
        for usuario_id, known_features_list in self.known_faces.items():
            for known_features in known_features_list:
                similarity = self.compare_features(features, known_features)
                
                if similarity > best_similarity:
                    second_best_similarity = best_similarity
                    best_similarity = similarity
                    best_match = usuario_id
                elif similarity > second_best_similarity:
                    second_best_similarity = similarity
        
        # Verifica se está acima do threshold (mais rigoroso)
        # Requer pelo menos 75% de similaridade E diferença significativa do segundo melhor
        if best_match and best_similarity >= self.threshold:
            # Verifica se há diferença significativa (pelo menos 5% melhor que o segundo)
            # Isso evita falsos positivos quando há múltiplas faces similares
            difference = best_similarity - second_best_similarity
            
            if best_similarity >= 0.75 and difference >= 0.05:
                usuario = self.database.buscar_usuario(best_match)
                if usuario:
                    return {
                        'usuario_id': int(best_match),
                        'nome': str(usuario['nome']),
                        'confidence': float(best_similarity),
                        'distance': float(1.0 - best_similarity)
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
                # Reconhece usando a imagem completa e o bbox
                recognition = self.recognize_face(image, (x, y, w, h))
                
                if recognition:
                    results.append({
                        'location': (int(x), int(y), int(w), int(h)),
                        'usuario_id': recognition['usuario_id'],
                        'nome': recognition['nome'],
                        'confidence': recognition['confidence']
                    })
                else:
                    results.append({
                        'location': (int(x), int(y), int(w), int(h)),
                        'usuario_id': None,
                        'nome': 'Desconhecido',
                        'confidence': 0.0
                    })
        
        return results
    
    def add_face(self, usuario_id: int, face_image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        Adiciona uma nova face ao sistema
        
        Args:
            usuario_id: ID do usuário
            face_image: Imagem do rosto ou imagem completa
            face_bbox: Opcional, bounding box do rosto
        """
        features = self.extract_features_from_array(face_image, face_bbox)
        if features is not None:
            if usuario_id not in self.known_faces:
                self.known_faces[usuario_id] = []
            self.known_faces[usuario_id].append(features)
