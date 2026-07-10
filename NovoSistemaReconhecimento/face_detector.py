"""
Módulo para detecção de rostos usando YOLOv8
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class FaceDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Inicializa o detector de rostos com YOLOv8
        model_path: caminho para o modelo YOLO (será baixado automaticamente se não existir)
        """
        self.model = YOLO(model_path)
        # Classe 'person' no COCO dataset é 0
        self.person_class = 0
    
    def detect_faces(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos em uma imagem usando YOLO para detectar pessoas
        e depois extrai a região superior (cabeça/rosto)
        
        Args:
            image: Imagem numpy array (BGR)
            conf_threshold: Threshold de confiança para detecção
            
        Returns:
            Lista de tuplas (x, y, width, height) representando bounding boxes dos rostos
        """
        # Executa detecção com YOLO
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Verifica se é uma pessoa
                cls = int(box.cls[0])
                if cls == self.person_class:
                    # Obtém coordenadas da bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extrai a região superior da pessoa (onde geralmente está o rosto)
                    # Usa aproximadamente os primeiros 30% da altura da pessoa
                    person_height = y2 - y1
                    face_height = int(person_height * 0.35)
                    
                    # Ajusta para garantir que não ultrapasse os limites
                    face_y1 = y1
                    face_y2 = min(y1 + face_height, y2)
                    
                    # Centraliza horizontalmente (rosto geralmente está no centro)
                    person_width = x2 - x1
                    face_width = int(person_width * 0.8)
                    face_x1 = x1 + int((person_width - face_width) / 2)
                    face_x2 = min(face_x1 + face_width, x2)
                    
                    faces.append((face_x1, face_y1, face_x2 - face_x1, face_y2 - face_y1))
        
        return faces
    
    def detect_faces_from_file(self, image_path: str, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostos em uma imagem a partir de um arquivo
        
        Args:
            image_path: Caminho para o arquivo de imagem
            conf_threshold: Threshold de confiança para detecção
            
        Returns:
            Lista de tuplas (x, y, width, height) representando bounding boxes dos rostos
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        return self.detect_faces(image, conf_threshold)
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extrai a região do rosto da imagem baseado na bounding box
        
        Args:
            image: Imagem numpy array (BGR)
            bbox: Tupla (x, y, width, height)
            
        Returns:
            Imagem do rosto extraída ou None se inválida
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
        return face_region
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                   labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Desenha bounding boxes dos rostos detectados na imagem
        
        Args:
            image: Imagem numpy array (BGR)
            faces: Lista de bounding boxes
            labels: Lista opcional de labels para cada rosto
            
        Returns:
            Imagem com bounding boxes desenhadas
        """
        result_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Desenha retângulo
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Adiciona label se fornecido
            if labels and i < len(labels):
                label = labels[i]
                # Calcula tamanho do texto
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                # Desenha fundo para o texto
                cv2.rectangle(
                    result_image,
                    (x, y - text_height - 10),
                    (x + text_width, y),
                    (0, 255, 0),
                    -1
                )
                # Desenha texto
                cv2.putText(
                    result_image,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
        
        return result_image
