"""
Módulo para análise facial avançada
Detecta óculos, olhos abertos/fechados, e outras características faciais
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class FacialAnalysis:
    def __init__(self):
        """Inicializa o analisador facial"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
    def detect_glasses(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Detecta se a pessoa está usando óculos
        
        Args:
            face_image: Imagem completa
            face_bbox: Bounding box do rosto (x, y, w, h)
            
        Returns:
            (has_glasses, confidence)
        """
        x, y, w, h = face_bbox
        
        # Extrai região do rosto
        face_region = face_image[y:y+h, x:x+w]
        if face_region.size == 0:
            return False, 0.0
        
        # Converte para escala de cinza
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region.copy()
        
        # Região dos olhos (parte superior do rosto)
        eye_region = gray_face[0:int(h*0.6), :]
        
        # 1. Tenta detectar olhos normais (sem óculos)
        eyes_normal = self.eye_cascade.detectMultiScale(
            eye_region, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(20, 20)
        )
        
        # 2. Tenta detectar olhos com óculos
        eyes_glasses = self.eye_glasses_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # 3. Análise de reflexos e bordas (óculos criam reflexos e bordas horizontais)
        # Detecta bordas horizontais na região dos olhos
        edges = cv2.Canny(eye_region, 50, 150)
        
        # Conta bordas horizontais (típicas de armações de óculos)
        horizontal_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        horizontal_edges = cv2.filter2D(edges, -1, horizontal_kernel)
        horizontal_edge_density = np.sum(horizontal_edges > 100) / eye_region.size
        
        # 4. Análise de reflexos (óculos criam reflexos brilhantes)
        bright_threshold = np.percentile(eye_region, 90)
        bright_pixels = np.sum(eye_region > bright_threshold)
        bright_ratio = bright_pixels / eye_region.size
        
        # 5. Análise de contraste (óculos aumentam contraste na região dos olhos)
        contrast = np.std(eye_region)
        
        # Lógica de decisão
        has_glasses = False
        confidence = 0.0
        
        # Se detectou olhos com óculos mas não olhos normais
        if len(eyes_glasses) >= 2 and len(eyes_normal) < 2:
            has_glasses = True
            confidence = 0.85
        # Se detectou olhos normais claramente
        elif len(eyes_normal) >= 2 and len(eyes_glasses) < 2:
            has_glasses = False
            confidence = 0.80
        # Caso ambíguo - usa análise de bordas e reflexos
        else:
            # Se há muitas bordas horizontais e reflexos, provavelmente tem óculos
            if horizontal_edge_density > 0.08 and bright_ratio > 0.05:
                has_glasses = True
                confidence = 0.70
            # Se há poucas bordas e poucos reflexos, provavelmente não tem óculos
            elif horizontal_edge_density < 0.03 and bright_ratio < 0.02:
                has_glasses = False
                confidence = 0.75
            # Caso intermediário
            else:
                # Se o contraste é muito alto, pode indicar óculos
                if contrast > 35:
                    has_glasses = True
                    confidence = 0.60
                else:
                    has_glasses = False
                    confidence = 0.55
        
        return has_glasses, confidence
    
    def detect_eye_state(self, eye_region: np.ndarray) -> Tuple[str, float]:
        """
        Detecta se o olho está aberto ou fechado
        
        Args:
            eye_region: Região do olho em escala de cinza
            
        Returns:
            (state, confidence) - state: 'open', 'closed', 'partial'
        """
        if eye_region.size == 0:
            return 'unknown', 0.0
        
        h, w = eye_region.shape
        
        # 1. Análise de altura vs largura (olhos fechados são mais horizontais)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # 2. Análise de variação vertical (olhos abertos têm mais variação vertical)
        # Calcula gradiente vertical
        grad_y = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
        vertical_variation = np.std(grad_y)
        
        # 3. Análise de intensidade (olhos fechados tendem a ser mais uniformes)
        intensity_std = np.std(eye_region)
        
        # 4. Análise de bordas horizontais (olhos fechados têm borda horizontal no meio)
        edges = cv2.Canny(eye_region, 30, 100)
        middle_row = edges[h//2, :]
        horizontal_edge_density = np.sum(middle_row > 0) / w
        
        # 5. Análise de área escura (pupila/íris em olhos abertos)
        dark_threshold = np.percentile(eye_region, 30)
        dark_pixels = np.sum(eye_region < dark_threshold)
        dark_ratio = dark_pixels / eye_region.size
        
        # Lógica de decisão
        open_score = 0.0
        closed_score = 0.0
        
        # Olhos abertos: alta variação vertical, área escura (pupila), aspecto mais vertical
        if aspect_ratio < 2.5:  # Mais vertical
            open_score += 0.3
        if vertical_variation > 15:
            open_score += 0.3
        if dark_ratio > 0.15:  # Tem área escura (pupila)
            open_score += 0.2
        if horizontal_edge_density < 0.3:  # Poucas bordas horizontais
            open_score += 0.2
        
        # Olhos fechados: baixa variação vertical, borda horizontal no meio, aspecto mais horizontal
        if aspect_ratio > 3.0:  # Mais horizontal
            closed_score += 0.3
        if vertical_variation < 10:
            closed_score += 0.3
        if horizontal_edge_density > 0.4:  # Muitas bordas horizontais
            closed_score += 0.2
        if intensity_std < 20:  # Mais uniforme
            closed_score += 0.2
        
        # Determina estado
        if open_score > closed_score + 0.2:
            return 'open', min(1.0, open_score)
        elif closed_score > open_score + 0.2:
            return 'closed', min(1.0, closed_score)
        else:
            return 'partial', 0.5
    
    def analyze_eyes(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Analisa estado dos olhos (aberto/fechado) e detecta óculos
        
        Args:
            face_image: Imagem completa
            face_bbox: Bounding box do rosto (x, y, w, h)
            
        Returns:
            Dicionário com informações sobre olhos e óculos
        """
        x, y, w, h = face_bbox
        
        # Extrai região do rosto
        face_region = face_image[y:y+h, x:x+w]
        if face_region.size == 0:
            return {
                'glasses': False,
                'glasses_confidence': 0.0,
                'left_eye': 'unknown',
                'right_eye': 'unknown',
                'left_eye_confidence': 0.0,
                'right_eye_confidence': 0.0
            }
        
        # Converte para escala de cinza
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region.copy()
        
        # Detecta óculos
        has_glasses, glasses_confidence = self.detect_glasses(face_image, face_bbox)
        
        # Região dos olhos
        eye_region_full = gray_face[0:int(h*0.65), :]
        
        # Tenta detectar olhos (com ou sem óculos)
        if has_glasses:
            eyes = self.eye_glasses_cascade.detectMultiScale(
                eye_region_full,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
        else:
            eyes = self.eye_cascade.detectMultiScale(
                eye_region_full,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        # Se não detectou, tenta o outro detector
        if len(eyes) < 2:
            if has_glasses:
                eyes = self.eye_cascade.detectMultiScale(
                    eye_region_full,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
            else:
                eyes = self.eye_glasses_cascade.detectMultiScale(
                    eye_region_full,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
        
        # Analisa estado dos olhos
        left_eye_state = 'unknown'
        right_eye_state = 'unknown'
        left_eye_conf = 0.0
        right_eye_conf = 0.0
        
        if len(eyes) >= 2:
            # Ordena olhos da esquerda para direita
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Olho esquerdo (do ponto de vista da pessoa)
            left_eye_bbox = eyes[0]
            try:
                # Valida índices para evitar erro de indexação
                ey, ex, ew, eh = left_eye_bbox
                if ey >= 0 and ex >= 0 and ey + eh <= eye_region_full.shape[0] and ex + ew <= eye_region_full.shape[1]:
                    left_eye_region = eye_region_full[
                        ey:ey+eh,
                        ex:ex+ew
                    ]
                    if left_eye_region.size > 0:
                        left_eye_state, left_eye_conf = self.detect_eye_state(left_eye_region)
                    else:
                        left_eye_state, left_eye_conf = 'unknown', 0.0
                else:
                    left_eye_state, left_eye_conf = 'unknown', 0.0
            except Exception as e:
                print(f"Erro ao processar olho esquerdo: {e}")
                left_eye_state, left_eye_conf = 'unknown', 0.0
            
            # Olho direito (do ponto de vista da pessoa)
            right_eye_bbox = eyes[1]
            try:
                # Valida índices para evitar erro de indexação
                ey, ex, ew, eh = right_eye_bbox
                if ey >= 0 and ex >= 0 and ey + eh <= eye_region_full.shape[0] and ex + ew <= eye_region_full.shape[1]:
                    right_eye_region = eye_region_full[
                        ey:ey+eh,
                        ex:ex+ew
                    ]
                    if right_eye_region.size > 0:
                        right_eye_state, right_eye_conf = self.detect_eye_state(right_eye_region)
                    else:
                        right_eye_state, right_eye_conf = 'unknown', 0.0
                else:
                    right_eye_state, right_eye_conf = 'unknown', 0.0
            except Exception as e:
                print(f"Erro ao processar olho direito: {e}")
                right_eye_state, right_eye_conf = 'unknown', 0.0
        elif len(eyes) == 1:
            # Só detectou um olho, analisa ele
            eye_bbox = eyes[0]
            try:
                ey, ex, ew, eh = eye_bbox
                if ey >= 0 and ex >= 0 and ey + eh <= eye_region_full.shape[0] and ex + ew <= eye_region_full.shape[1]:
                    eye_region = eye_region_full[
                        ey:ey+eh,
                        ex:ex+ew
                    ]
                    if eye_region.size > 0:
                        state, conf = self.detect_eye_state(eye_region)
                    else:
                        state, conf = 'unknown', 0.0
                else:
                    state, conf = 'unknown', 0.0
            except Exception as e:
                print(f"Erro ao processar olho único: {e}")
                state, conf = 'unknown', 0.0
            left_eye_state = state
            right_eye_state = state
            left_eye_conf = conf
            right_eye_conf = conf
        
        return {
            'glasses': has_glasses,
            'glasses_confidence': float(glasses_confidence),
            'left_eye': left_eye_state,
            'right_eye': right_eye_state,
            'left_eye_confidence': float(left_eye_conf),
            'right_eye_confidence': float(right_eye_conf),
            'eyes_detected': len(eyes)
        }
    
    def analyze_full_face(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Análise completa do rosto incluindo óculos, olhos, e outras características
        
        Args:
            face_image: Imagem completa
            face_bbox: Bounding box do rosto (x, y, w, h)
            
        Returns:
            Dicionário com todas as características detectadas
        """
        analysis = self.analyze_eyes(face_image, face_bbox)
        
        # Adiciona informações adicionais
        x, y, w, h = face_bbox
        face_region = face_image[y:y+h, x:x+w]
        
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region.copy()
        
        # Análise de iluminação
        mean_brightness = np.mean(gray_face)
        brightness_std = np.std(gray_face)
        
        # Análise de qualidade
        # Calcula nitidez (variação de gradiente)
        grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        analysis.update({
            'brightness': float(mean_brightness),
            'brightness_std': float(brightness_std),
            'sharpness': float(sharpness),
            'quality': 'good' if sharpness > 30 and 50 < mean_brightness < 200 else 'poor'
        })
        
        return analysis
