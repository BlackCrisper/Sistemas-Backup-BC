"""
Módulo para reconhecimento facial usando face_recognition
"""
import face_recognition
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from database import Database


class FaceRecognizer:
    def __init__(self, database: Database, tolerance: float = 0.6):
        """
        Inicializa o reconhecedor facial
        
        Args:
            database: Instância do Database para buscar encodings
            tolerance: Tolerância para comparação (menor = mais rigoroso, padrão 0.6)
        """
        self.database = database
        self.tolerance = tolerance
    
    def extract_encoding(self, image: np.ndarray, face_location: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Extrai encoding facial de uma imagem
        
        Args:
            image: Imagem numpy array (RGB ou BGR)
            face_location: Opcional, localização do rosto (top, right, bottom, left)
                          Se None, detecta automaticamente
            
        Returns:
            Encoding numpy array ou None se nenhum rosto for encontrado
        """
        # Converte BGR para RGB se necessário
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Verifica se é BGR (OpenCV) ou RGB
            # face_recognition espera RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        try:
            if face_location:
                # Usa localização fornecida
                top, right, bottom, left = face_location
                # face_recognition usa formato (top, right, bottom, left)
                # mas pode receber (y, x+w, y+h, x) se vier de bounding box (x, y, w, h)
                if len(face_location) == 4:
                    # Assume formato (x, y, w, h) e converte
                    x, y, w, h = face_location
                    top, right, bottom, left = y, x + w, y + h, x
                
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    [(top, right, bottom, left)]
                )
            else:
                # Detecta automaticamente
                encodings = face_recognition.face_encodings(rgb_image)
            
            if encodings:
                return encodings[0]  # Retorna o primeiro encoding encontrado
            return None
        except Exception as e:
            print(f"Erro ao extrair encoding: {e}")
            return None
    
    def extract_encodings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extrai todos os encodings faciais de uma imagem
        
        Args:
            image: Imagem numpy array (RGB ou BGR)
            
        Returns:
            Lista de encodings numpy arrays
        """
        # Converte BGR para RGB se necessário
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        try:
            encodings = face_recognition.face_encodings(rgb_image)
            return encodings
        except Exception as e:
            print(f"Erro ao extrair encodings: {e}")
            return []
    
    def compare_face(self, encoding: np.ndarray, known_encoding: np.ndarray) -> bool:
        """
        Compara dois encodings faciais
        
        Args:
            encoding: Encoding a ser comparado
            known_encoding: Encoding conhecido
            
        Returns:
            True se as faces são similares, False caso contrário
        """
        distance = face_recognition.face_distance([known_encoding], encoding)[0]
        return distance <= self.tolerance
    
    def recognize_face(self, encoding: np.ndarray) -> Optional[Dict]:
        """
        Reconhece uma face comparando com o banco de dados
        
        Args:
            encoding: Encoding da face a ser reconhecida
            
        Returns:
            Dicionário com informações do usuário reconhecido ou None
            {'usuario_id': int, 'nome': str, 'confidence': float}
        """
        # Busca todos os encodings do banco
        known_encodings = self.database.buscar_todos_encodings()
        
        if not known_encodings:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for known_data in known_encodings:
            known_encoding = known_data['encoding']
            distance = face_recognition.face_distance([known_encoding], encoding)[0]
            
            if distance < best_distance:
                best_distance = distance
                best_match = known_data
        
        # Verifica se a distância está dentro da tolerância
        if best_match and best_distance <= self.tolerance:
            # Calcula confiança (1 - distância normalizada)
            confidence = max(0, 1 - (best_distance / self.tolerance))
            
            return {
                'usuario_id': best_match['usuario_id'],
                'nome': best_match['nome'],
                'confidence': confidence,
                'distance': best_distance
            }
        
        return None
    
    def recognize_faces(self, encodings: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Reconhece múltiplas faces
        
        Args:
            encodings: Lista de encodings faciais
            
        Returns:
            Lista de resultados de reconhecimento (pode conter None)
        """
        results = []
        for encoding in encodings:
            result = self.recognize_face(encoding)
            results.append(result)
        return results
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta e reconhece todas as faces em uma imagem
        
        Args:
            image: Imagem numpy array (BGR)
            
        Returns:
            Lista de dicionários com informações das faces reconhecidas
            Cada dicionário contém: 'location', 'usuario_id', 'nome', 'confidence'
        """
        # Converte para RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detecta localizações das faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        # Extrai encodings
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = []
        for i, encoding in enumerate(encodings):
            recognition = self.recognize_face(encoding)
            
            # Converte localização para formato (x, y, w, h)
            top, right, bottom, left = face_locations[i]
            location = (left, top, right - left, bottom - top)
            
            if recognition:
                results.append({
                    'location': location,
                    'usuario_id': recognition['usuario_id'],
                    'nome': recognition['nome'],
                    'confidence': recognition['confidence']
                })
            else:
                results.append({
                    'location': location,
                    'usuario_id': None,
                    'nome': 'Desconhecido',
                    'confidence': 0.0
                })
        
        return results
