"""
Módulo para detecção de vivacidade (liveness detection)
Detecta se é uma pessoa real ou uma foto/impressão
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import time


class LivenessDetector:
    def __init__(self):
        """Inicializa o detector de vivacidade"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Histórico de frames para análise de movimento
        self.frame_history = []
        self.max_history = 5
        
    def detect_photo_artifacts(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detecta artefatos que indicam que é uma foto impressa ou em tela
        
        Returns:
            (is_real, confidence, reason)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        score = 1.0  # Começa assumindo que é real
        reasons = []
        
        # 1. Detecta bordas muito definidas (foto impressa tem bordas nítidas)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Fotos impressas tendem a ter muitas bordas nítidas
        if edge_density > 0.15:  # Muitas bordas podem indicar foto
            score -= 0.15
            reasons.append("Muitas bordas nítidas detectadas")
        
        # 2. Analisa variação de brilho (fotos têm menos variação natural)
        # Calcula desvio padrão em blocos
        block_size = 32
        std_values = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                std_values.append(np.std(block))
        
        mean_std = np.mean(std_values) if std_values else 0
        
        # Fotos impressas têm menos variação natural de brilho
        if mean_std < 15:  # Muito uniforme pode ser foto
            score -= 0.20
            reasons.append("Variação de brilho muito baixa")
        
        # 3. Detecta padrões de impressão (moiré, textura de papel)
        # Aplica filtro de frequência para detectar padrões repetitivos
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Padrões de impressão criam picos no espectro de frequência
        # Verifica se há picos muito pronunciados (indicam padrão repetitivo)
        threshold = np.percentile(magnitude_spectrum, 95)
        high_freq_pixels = np.sum(magnitude_spectrum > threshold)
        high_freq_ratio = high_freq_pixels / (h * w)
        
        if high_freq_ratio > 0.10:  # Muitos picos podem indicar padrão de impressão
            score -= 0.15
            reasons.append("Padrões de impressão detectados")
        
        # 4. Analisa reflexos nos olhos (pessoas reais têm reflexos naturais)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            eye_region = gray[y:y+int(h*0.5), x:x+w]
            
            if eye_region.size > 0:
                # Olhos reais têm reflexos (pontos brilhantes)
                # Procura por pontos muito brilhantes na região dos olhos
                bright_threshold = np.percentile(eye_region, 95)
                bright_pixels = np.sum(eye_region > bright_threshold)
                bright_ratio = bright_pixels / eye_region.size
                
                # Reflexos naturais aparecem como pequenas áreas muito brilhantes
                if bright_ratio < 0.02:  # Muito poucos reflexos pode ser foto
                    score -= 0.10
                    reasons.append("Poucos reflexos nos olhos")
        
        # 5. Detecta textura de papel (superfície muito uniforme)
        # Calcula variância local
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        mean_variance = np.mean(local_variance)
        
        # Fotos impressas têm variância muito baixa
        if mean_variance < 100:
            score -= 0.15
            reasons.append("Textura muito uniforme (possível foto)")
        
        # 6. Detecta se é uma tela (padrões de pixelização)
        # Imagens de tela têm padrões de subpixels
        # Calcula autocorrelação para detectar padrões repetitivos
        if h > 100 and w > 100:
            sample = gray[0:100, 0:100]
            autocorr = cv2.matchTemplate(gray, sample, cv2.TM_CCOEFF_NORMED)
            max_corr = np.max(autocorr)
            
            # Padrões muito repetitivos podem indicar tela
            if max_corr > 0.7:
                score -= 0.10
                reasons.append("Padrões repetitivos detectados (possível tela)")
        
        # Normaliza score para [0, 1]
        score = max(0.0, min(1.0, score))
        
        # Considera real se score > 0.5
        is_real = score > 0.5
        reason = "; ".join(reasons) if reasons else "OK"
        
        return is_real, score, reason
    
    def detect_movement(self, current_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detecta movimento entre frames (pessoas reais se movem)
        
        Returns:
            (has_movement, movement_score)
        """
        if len(current_frame.shape) == 3:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = current_frame.copy()
        
        # Adiciona frame ao histórico
        self.frame_history.append(gray.copy())
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Precisa de pelo menos 2 frames para detectar movimento
        if len(self.frame_history) < 2:
            return True, 0.5  # Assume movimento se não tem histórico suficiente
        
        # Compara com frame anterior
        prev_frame = self.frame_history[-2]
        current_frame_resized = cv2.resize(gray, (prev_frame.shape[1], prev_frame.shape[0]))
        
        # Calcula diferença
        diff = cv2.absdiff(prev_frame, current_frame_resized)
        movement_score = np.sum(diff > 30) / diff.size  # Porcentagem de pixels que mudaram
        
        # Se houver movimento significativo (>1% dos pixels), é provável que seja real
        has_movement = movement_score > 0.01
        
        return has_movement, float(movement_score)
    
    def detect_blink(self, eye_region: np.ndarray) -> bool:
        """
        Detecta piscar de olhos (indicador de pessoa real)
        Nota: Requer múltiplos frames para funcionar bem
        """
        # Análise básica: olhos fechados têm menos variação
        if eye_region.size == 0:
            return False
        
        std = np.std(eye_region)
        mean = np.mean(eye_region)
        
        # Olhos fechados tendem a ter menos variação
        # Esta é uma detecção simplificada
        return std < mean * 0.3
    
    def validate_liveness(self, image: np.ndarray, require_movement: bool = False) -> Tuple[bool, str, float]:
        """
        Valida se é uma pessoa real (não uma foto)
        
        Args:
            image: Imagem para validar
            require_movement: Se True, requer movimento detectado
            
        Returns:
            (is_real, message, confidence)
        """
        # 1. Detecta artefatos de foto
        is_real_photo, photo_score, photo_reason = self.detect_photo_artifacts(image)
        
        # 2. Detecta movimento (se histórico disponível)
        has_movement, movement_score = self.detect_movement(image)
        
        # Combina resultados
        if require_movement and not has_movement:
            return False, "Nenhum movimento detectado. Por favor, mova-se levemente.", 0.0
        
        # Score combinado
        if require_movement:
            combined_score = 0.6 * photo_score + 0.4 * (1.0 if has_movement else 0.0)
        else:
            combined_score = photo_score
        
        # Considera real se score > 0.45 (mais permissivo para webcam)
        is_real = combined_score > 0.45
        
        if not is_real:
            message = f"Possível foto detectada: {photo_reason}"
            if require_movement and not has_movement:
                message += " Sem movimento detectado."
        else:
            message = "Pessoa real detectada"
            if photo_reason != "OK":
                message += f" ({photo_reason})"
        
        return is_real, message, float(combined_score)
    
    def reset_history(self):
        """Reseta o histórico de frames"""
        self.frame_history = []
