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

    def detect_sunglasses(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                          landmarks: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Detecta óculos escuros (lentes opacas). Conservador para evitar falso positivo
        com sombra, sobrancelha ou íris escura. Óculos transparentes não bloqueiam.
        """
        x, y, w, h = face_bbox
        if w <= 0 or h <= 0:
            return False, 0.0

        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        patches = []
        if landmarks and 'left_eye' in landmarks and 'right_eye' in landmarks:
            for key in ('left_eye', 'right_eye'):
                # Patch um pouco menor: foca na lente, não na sobrancelha
                patch = self._eye_patch_from_landmark(
                    gray, landmarks[key], max(1, int(w * 0.85)), max(1, int(h * 0.75))
                )
                if patch is not None and patch.size > 0:
                    patches.append(patch)

        if len(patches) < 2:
            eye_band = gray[y + int(h * 0.22):y + int(h * 0.42), x + int(w * 0.15):x + int(w * 0.85)]
            if eye_band.size == 0:
                return False, 0.0
            mid = eye_band.shape[1] // 2
            patches = [eye_band[:, :mid], eye_band[:, mid:]]

        cheek = gray[y + int(h * 0.55):y + int(h * 0.75), x + int(w * 0.25):x + int(w * 0.75)]
        forehead = gray[y + int(h * 0.08):y + int(h * 0.22), x + int(w * 0.25):x + int(w * 0.75)]
        cheek_mean = float(np.mean(cheek)) if cheek.size else 120.0
        forehead_mean = float(np.mean(forehead)) if forehead.size else cheek_mean
        skin_ref = max(cheek_mean, forehead_mean)

        per_eye = []
        for patch in patches[:2]:
            if patch.size == 0:
                continue
            mean_v = float(np.mean(patch))
            std_v = float(np.std(patch))
            dark_ratio = float(np.mean(patch < 40))
            # Íris/pupila visível → textura; lente opaca → escuro e homogêneo
            opaque = (
                mean_v < skin_ref - 55
                and mean_v < 48
                and dark_ratio > 0.55
                and std_v < 16
            )
            per_eye.append({
                'opaque': opaque,
                'mean': mean_v,
                'std': std_v,
                'dark_ratio': dark_ratio,
            })

        if len(per_eye) < 2:
            return False, 0.0

        # Ambos os olhos precisam parecer lente opaca (evita sombra só de um lado)
        if not (per_eye[0]['opaque'] and per_eye[1]['opaque']):
            return False, 0.0

        # Se ainda há bastante textura média, provavelmente são olhos reais
        if float(np.mean([e['std'] for e in per_eye])) >= 18:
            return False, 0.0

        score = 0.55
        mean_gap = skin_ref - float(np.mean([e['mean'] for e in per_eye]))
        if mean_gap > 65:
            score += 0.2
        if float(np.mean([e['dark_ratio'] for e in per_eye])) > 0.7:
            score += 0.15
        if float(np.mean([e['std'] for e in per_eye])) < 12:
            score += 0.1

        has_sunglasses = score >= 0.75
        return has_sunglasses, float(min(1.0, score))
    
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
        Detecta chapéu/boné de forma conservadora.
        Cabelo escuro acima da testa NÃO deve contar como chapéu.
        Exige evidência de aba/cobertura opaca sobre a testa.
        """
        x, y, w, h = face_bbox
        img_h, img_w = face_image.shape[:2]

        # Faixa logo acima do topo do bbox (possível aba)
        brim_h = max(10, int(h * 0.22))
        y1 = max(0, y - brim_h)
        y2 = max(0, min(img_h, y + int(h * 0.06)))
        x1 = max(0, x + int(w * 0.12))
        x2 = min(img_w, x + int(w * 0.88))

        # Testa (deve ser pele se não houver boné baixo)
        fh_y1 = y + int(h * 0.08)
        fh_y2 = y + int(h * 0.28)
        fh_x1 = x + int(w * 0.25)
        fh_x2 = x + int(w * 0.75)

        face_mid = face_image[
            y + int(h * 0.35):y + int(h * 0.55),
            x + int(w * 0.25):x + int(w * 0.75)
        ]
        if y2 <= y1 or x2 <= x1 or face_mid.size == 0:
            return False, 0.0
        if fh_y2 <= fh_y1 or fh_x2 <= fh_x1:
            return False, 0.0

        top = face_image[y1:y2, x1:x2]
        forehead = face_image[fh_y1:fh_y2, fh_x1:fh_x2]
        if top.size == 0 or forehead.size == 0:
            return False, 0.0

        if len(top.shape) == 3:
            top_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
            mid_gray = cv2.cvtColor(face_mid, cv2.COLOR_BGR2GRAY)
            fh_gray = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY)
            top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            mid_hsv = cv2.cvtColor(face_mid, cv2.COLOR_BGR2HSV)
            fh_hsv = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
        else:
            top_gray = top
            mid_gray = face_mid
            fh_gray = forehead
            top_hsv = mid_hsv = fh_hsv = None

        top_mean = float(np.mean(top_gray))
        mid_mean = float(np.mean(mid_gray))
        fh_mean = float(np.mean(fh_gray))
        brightness_gap = mid_mean - top_mean
        forehead_gap = mid_mean - fh_mean

        # Aba/boné tende a ser mais uniforme que cabelo
        top_std = float(np.std(top_gray))
        fh_std = float(np.std(fh_gray))

        # Borda horizontal forte na linha da testa (aba do boné)
        edge_band = top_gray[-max(3, top_gray.shape[0] // 3):, :]
        edges = cv2.Canny(edge_band, 60, 140)
        horizontal_edge = float(np.mean(edges > 0))

        sat_gap = 0.0
        if top_hsv is not None and mid_hsv is not None:
            sat_gap = float(np.mean(top_hsv[:, :, 1]) - np.mean(mid_hsv[:, :, 1]))

        # Testa coberta: muito mais escura que a face e com pouca variação de pele
        forehead_covered = forehead_gap > 40 and fh_mean < 70 and fh_std < 22
        strong_brim = (
            brightness_gap > 45
            and top_mean < 60
            and top_std < 28
            and horizontal_edge > 0.12
        )

        score = 0.0
        if strong_brim:
            score += 0.45
        if forehead_covered:
            score += 0.40
        if sat_gap > 40 and brightness_gap > 40:
            score += 0.15

        # Cabelo sozinho (textura alta, sem cobertura da testa) → não é chapéu
        if not forehead_covered and top_std > 32 and horizontal_edge < 0.08:
            return False, float(min(0.4, score))

        has_hat = score >= 0.75
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
                'sunglasses': False,
                'sunglasses_confidence': 0.0,
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
        has_sunglasses, sunglasses_confidence = self.detect_sunglasses(
            face_image, face_bbox, landmarks=landmarks
        )

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

        # Óculos escuros impedem confirmação confiável dos olhos
        if has_sunglasses:
            eyes_open = False

        return {
            'glasses': bool(has_glasses),
            'glasses_confidence': float(glasses_confidence),
            'sunglasses': bool(has_sunglasses),
            'sunglasses_confidence': float(sunglasses_confidence),
            'left_eye': left_eye_state,
            'right_eye': right_eye_state,
            'left_eye_confidence': float(left_eye_conf),
            'right_eye_confidence': float(right_eye_conf),
            'eyes_open': bool(eyes_open),
            'eyes_detected': int(eyes_count),
        }

    def validate_frontal_pose(self, landmarks: Optional[Dict]) -> Tuple[bool, str]:
        """Verifica pose frontal aproximada via landmarks YuNet."""
        if not landmarks:
            return False, 'pose não confirmada'

        required = ('left_eye', 'right_eye', 'nose')
        if not all(k in landmarks for k in required):
            return False, 'pose não confirmada'

        lx, ly = landmarks['left_eye']
        rx, ry = landmarks['right_eye']
        nx, ny = landmarks['nose']

        eye_dx = abs(float(lx) - float(rx))
        if eye_dx < 1e-3:
            return False, 'olhe para a câmera'

        eye_dy_ratio = abs(float(ly) - float(ry)) / eye_dx
        if eye_dy_ratio > 0.22:
            return False, 'mantenha a cabeça reta'

        mid_x = (float(lx) + float(rx)) / 2.0
        nose_offset = abs(float(nx) - mid_x) / eye_dx
        if nose_offset > 0.28:
            return False, 'vire o rosto de frente'

        eye_y = (float(ly) + float(ry)) / 2.0
        if float(ny) < eye_y:
            return False, 'olhe para a câmera'

        return True, 'OK'

    def analyze_full_face(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                          landmarks: Optional[Dict] = None,
                          frame_shape: Optional[Tuple[int, ...]] = None) -> Dict:
        """
        Análise completa com checklist de escaneamento para cadastro.
        """
        analysis = self.analyze_eyes(face_image, face_bbox, landmarks=landmarks)
        has_hat, hat_confidence = self.detect_hat(face_image, face_bbox)

        x, y, w, h = face_bbox
        face_region = face_image[y:y + h, x:x + w]

        def empty_result(blockers):
            checks = [
                {'id': 'face', 'label': 'Rosto detectado', 'ok': False, 'detail': 'rosto inválido'},
                {'id': 'lighting', 'label': 'Iluminação', 'ok': False, 'detail': '—'},
                {'id': 'sharpness', 'label': 'Nitidez', 'ok': False, 'detail': '—'},
                {'id': 'eyes', 'label': 'Olhos abertos', 'ok': False, 'detail': '—'},
                {'id': 'sunglasses', 'label': 'Sem óculos escuros', 'ok': False, 'detail': '—'},
                {'id': 'hat', 'label': 'Sem chapéu/boné', 'ok': False, 'detail': '—'},
                {'id': 'pose', 'label': 'Pose de frente', 'ok': False, 'detail': '—'},
                {'id': 'position', 'label': 'Posição na câmera', 'ok': False, 'detail': '—'},
            ]
            analysis.update({
                'hat': False,
                'hat_confidence': 0.0,
                'sunglasses': bool(analysis.get('sunglasses', False)),
                'sunglasses_confidence': float(analysis.get('sunglasses_confidence', 0.0)),
                'accessories': [],
                'brightness': 0.0,
                'sharpness': 0.0,
                'quality': 'poor',
                'pose_ok': False,
                'position_ok': False,
                'checks': checks,
                'scan_progress': 0.0,
                'capture_ready': False,
                'capture_blockers': blockers,
            })
            return analysis

        if face_region.size == 0:
            return empty_result(['rosto inválido'])

        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region.copy()

        mean_brightness = float(np.mean(gray_face))
        grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))

        has_sunglasses = bool(analysis.get('sunglasses', False))
        eyes_open = bool(analysis.get('eyes_open', False))
        lighting_ok = 45.0 <= mean_brightness <= 210.0
        sharpness_ok = sharpness >= 18.0
        pose_ok, pose_detail = self.validate_frontal_pose(landmarks)

        # Posição relativa ao frame (quando dimensões disponíveis)
        position_ok = True
        position_detail = 'OK'
        if frame_shape is not None and len(frame_shape) >= 2:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
            if frame_h > 0 and frame_w > 0:
                cx = x + w / 2.0
                cy = y + h / 2.0
                offset_x = abs(cx - frame_w / 2.0) / frame_w
                offset_y = abs(cy - frame_h / 2.0) / frame_h
                height_ratio = h / float(frame_h)
                if offset_x > 0.18 or offset_y > 0.18:
                    position_ok = False
                    position_detail = 'centralize o rosto'
                elif height_ratio < 0.22:
                    position_ok = False
                    position_detail = 'aproxime-se'
                elif height_ratio > 0.62:
                    position_ok = False
                    position_detail = 'afaste-se um pouco'
                else:
                    position_detail = 'OK'

        accessories = []
        if has_sunglasses:
            accessories.append('óculos escuros')
        elif analysis.get('glasses'):
            accessories.append('óculos')
        if has_hat:
            accessories.append('chapéu/boné')

        if not lighting_ok:
            lighting_detail = 'pouca luz' if mean_brightness < 45 else 'muita luz'
        else:
            lighting_detail = 'OK'

        if eyes_open:
            eyes_detail = 'OK'
        elif analysis.get('left_eye') == 'closed' or analysis.get('right_eye') == 'closed':
            eyes_detail = 'abra os olhos'
        else:
            eyes_detail = 'olhos não confirmados'

        checks = [
            {'id': 'face', 'label': 'Rosto detectado', 'ok': True, 'detail': 'OK'},
            {
                'id': 'position',
                'label': 'Posição na câmera',
                'ok': bool(position_ok),
                'detail': position_detail,
            },
            {
                'id': 'lighting',
                'label': 'Iluminação',
                'ok': bool(lighting_ok),
                'detail': lighting_detail,
            },
            {
                'id': 'sharpness',
                'label': 'Nitidez',
                'ok': bool(sharpness_ok),
                'detail': 'OK' if sharpness_ok else 'imagem borrada',
            },
            {
                'id': 'eyes',
                'label': 'Olhos abertos',
                'ok': bool(eyes_open) and not has_sunglasses,
                'detail': 'óculos escuros' if has_sunglasses else eyes_detail,
            },
            {
                'id': 'sunglasses',
                'label': 'Sem óculos escuros',
                'ok': not has_sunglasses,
                'detail': 'remova os óculos escuros' if has_sunglasses else 'OK',
            },
            {
                'id': 'hat',
                'label': 'Sem chapéu/boné',
                'ok': not bool(has_hat),
                'detail': 'remova chapéu/boné' if has_hat else 'OK',
            },
            {
                'id': 'pose',
                'label': 'Pose de frente',
                'ok': bool(pose_ok),
                'detail': pose_detail if not pose_ok else 'OK',
            },
        ]

        blockers = [c['detail'] for c in checks if not c['ok'] and c['id'] != 'face']
        ok_count = sum(1 for c in checks if c['ok'])
        scan_progress = float(ok_count) / float(len(checks)) if checks else 0.0
        capture_ready = all(c['ok'] for c in checks)

        analysis.update({
            'hat': bool(has_hat),
            'hat_confidence': float(hat_confidence),
            'sunglasses': has_sunglasses,
            'sunglasses_confidence': float(analysis.get('sunglasses_confidence', 0.0)),
            'accessories': accessories,
            'brightness': mean_brightness,
            'brightness_std': float(np.std(gray_face)),
            'sharpness': sharpness,
            'quality': 'good' if sharpness_ok and lighting_ok else 'poor',
            'pose_ok': bool(pose_ok),
            'position_ok': bool(position_ok),
            'checks': checks,
            'scan_progress': scan_progress,
            'capture_ready': capture_ready,
            'capture_blockers': blockers,
        })

        return analysis
