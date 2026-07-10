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
    
    def detect_hat(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Heurística para chapéu/boné: região acima da testa com textura/cor diferente da pele.
        """
        x, y, w, h = face_bbox
        img_h, img_w = face_image.shape[:2]

        # Faixa acima do rosto (onde ficaria o chapéu)
        band_h = max(12, int(h * 0.35))
        y1 = max(0, y - band_h)
        y2 = max(0, y + int(h * 0.12))
        x1 = max(0, x + int(w * 0.1))
        x2 = min(img_w, x + int(w * 0.9))

        if y2 <= y1 or x2 <= x1:
            return False, 0.0

        top = face_image[y1:y2, x1:x2]
        face_mid = face_image[y + int(h * 0.25):y + int(h * 0.55), x + int(w * 0.25):x + int(w * 0.75)]
        if top.size == 0 or face_mid.size == 0:
            return False, 0.0

        if len(top.shape) == 3:
            top_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
            mid_gray = cv2.cvtColor(face_mid, cv2.COLOR_BGR2GRAY)
            top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            mid_hsv = cv2.cvtColor(face_mid, cv2.COLOR_BGR2HSV)
        else:
            top_gray = top
            mid_gray = face_mid
            top_hsv = None
            mid_hsv = None

        # Diferença de brilho e saturação (chapéu costuma ser mais escuro/saturado que a pele)
        brightness_gap = float(np.mean(mid_gray) - np.mean(top_gray))
        texture_top = float(np.std(top_gray))
        texture_mid = float(np.std(mid_gray))

        sat_gap = 0.0
        if top_hsv is not None and mid_hsv is not None:
            sat_gap = float(np.mean(top_hsv[:, :, 1]) - np.mean(mid_hsv[:, :, 1]))

        # Rosto muito baixo no frame (espaço grande acima) também sugere chapéu/boné
        top_margin_ratio = y / max(img_h, 1)

        score = 0.0
        if brightness_gap > 32:
            score += 0.35
        if sat_gap > 25:
            score += 0.25
        if texture_top > texture_mid * 1.25 and texture_top > 28:
            score += 0.2
        if top_margin_ratio > 0.28 and brightness_gap > 20:
            score += 0.2

        has_hat = score >= 0.55
        return has_hat, float(min(1.0, score))

    def _eye_patch_from_landmark(self, gray: np.ndarray, point: Tuple[float, float], face_w: int, face_h: int) -> Optional[np.ndarray]:
        px, py = int(point[0]), int(point[1])
        ew = max(12, int(face_w * 0.18))
        eh = max(10, int(face_h * 0.12))
        x1 = max(0, px - ew // 2)
        y1 = max(0, py - eh // 2)
        x2 = min(gray.shape[1], x1 + ew)
        y2 = min(gray.shape[0], y1 + eh)
        if x2 <= x1 or y2 <= y1:
            return None
        return gray[y1:y2, x1:x2]

    def analyze_eyes(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                     landmarks: Optional[Dict] = None) -> Dict:
        """
        Analisa estado dos olhos (aberto/fechado) e detecta óculos
        """
        x, y, w, h = face_bbox

        face_region = face_image[y:y+h, x:x+w]
        if face_region.size == 0:
            return {
                'glasses': False,
                'glasses_confidence': 0.0,
                'left_eye': 'unknown',
                'right_eye': 'unknown',
                'left_eye_confidence': 0.0,
                'right_eye_confidence': 0.0,
                'eyes_open': False,
                'eyes_detected': 0,
            }

        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region.copy()

        if len(face_image.shape) == 3:
            gray_full = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_full = face_image.copy()

        has_glasses, glasses_confidence = self.detect_glasses(face_image, face_bbox)

        left_eye_state = 'unknown'
        right_eye_state = 'unknown'
        left_eye_conf = 0.0
        right_eye_conf = 0.0
        eyes_count = 0

        # Preferência: patches ao redor dos landmarks (YuNet)
        if landmarks and 'left_eye' in landmarks and 'right_eye' in landmarks:
            left_patch = self._eye_patch_from_landmark(gray_full, landmarks['left_eye'], w, h)
            right_patch = self._eye_patch_from_landmark(gray_full, landmarks['right_eye'], w, h)
            if left_patch is not None and left_patch.size > 0:
                left_eye_state, left_eye_conf = self.detect_eye_state(left_patch)
                eyes_count += 1
            if right_patch is not None and right_patch.size > 0:
                right_eye_state, right_eye_conf = self.detect_eye_state(right_patch)
                eyes_count += 1

        # Fallback: cascade de olhos
        if eyes_count < 2:
            eye_region_full = gray_face[0:int(h * 0.65), :]
            eyes = self.eye_glasses_cascade.detectMultiScale(
                eye_region_full, 1.1, 3, minSize=(18, 18)
            ) if has_glasses else self.eye_cascade.detectMultiScale(
                eye_region_full, 1.1, 3, minSize=(18, 18)
            )
            if len(eyes) < 2:
                eyes = self.eye_cascade.detectMultiScale(eye_region_full, 1.1, 3, minSize=(18, 18))

            if len(eyes) >= 1:
                eyes = sorted(eyes, key=lambda e: e[0])
                eyes_count = len(eyes)
                for idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                    patch = eye_region_full[ey:ey + eh, ex:ex + ew]
                    if patch.size == 0:
                        continue
                    state, conf = self.detect_eye_state(patch)
                    if idx == 0:
                        left_eye_state, left_eye_conf = state, conf
                    else:
                        right_eye_state, right_eye_conf = state, conf
                if len(eyes) == 1:
                    right_eye_state, right_eye_conf = left_eye_state, left_eye_conf

        eyes_open = (
            left_eye_state == 'open' and right_eye_state == 'open'
        ) or (
            # Aceita um open + partial se confiança razoável
            {left_eye_state, right_eye_state} <= {'open', 'partial'}
            and 'open' in (left_eye_state, right_eye_state)
            and min(left_eye_conf, right_eye_conf) >= 0.45
        )

        # Se ambos unknown, não libera captura
        if left_eye_state == 'unknown' and right_eye_state == 'unknown':
            eyes_open = False

        return {
            'glasses': bool(has_glasses),
            'glasses_confidence': float(glasses_confidence),
            'left_eye': left_eye_state,
            'right_eye': right_eye_state,
            'left_eye_confidence': float(left_eye_conf),
            'right_eye_confidence': float(right_eye_conf),
            'eyes_open': bool(eyes_open),
            'eyes_detected': int(eyes_count),
        }

    def analyze_full_face(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                          landmarks: Optional[Dict] = None) -> Dict:
        """
        Análise completa: olhos, óculos, chapéu e qualidade.
        """
        analysis = self.analyze_eyes(face_image, face_bbox, landmarks=landmarks)
        has_hat, hat_confidence = self.detect_hat(face_image, face_bbox)

        x, y, w, h = face_bbox
        face_region = face_image[y:y + h, x:x + w]

        if face_region.size == 0:
            analysis.update({
                'hat': False,
                'hat_confidence': 0.0,
                'accessories': [],
                'brightness': 0.0,
                'sharpness': 0.0,
                'quality': 'poor',
                'capture_ready': False,
                'capture_blockers': ['rosto inválido'],
            })
            return analysis

        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region.copy()

        mean_brightness = float(np.mean(gray_face))
        grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))

        accessories = []
        if analysis.get('glasses'):
            accessories.append('óculos')
        if has_hat:
            accessories.append('chapéu/boné')

        blockers = []
        if not analysis.get('eyes_open'):
            if analysis.get('left_eye') == 'closed' or analysis.get('right_eye') == 'closed':
                blockers.append('abra os olhos')
            else:
                blockers.append('olhos não confirmados')
        if mean_brightness < 40:
            blockers.append('pouca luz')
        if mean_brightness > 220:
            blockers.append('muita luz')
        if sharpness < 18:
            blockers.append('imagem borrada')

        analysis.update({
            'hat': bool(has_hat),
            'hat_confidence': float(hat_confidence),
            'accessories': accessories,
            'brightness': mean_brightness,
            'brightness_std': float(np.std(gray_face)),
            'sharpness': sharpness,
            'quality': 'good' if sharpness > 30 and 50 < mean_brightness < 200 else 'poor',
            'capture_ready': len(blockers) == 0 and analysis.get('eyes_open', False),
            'capture_blockers': blockers,
        })

        return analysis
