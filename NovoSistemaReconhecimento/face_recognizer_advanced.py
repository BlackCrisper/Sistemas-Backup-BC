"""
Reconhecimento facial com OpenCV YuNet (detecção) + SFace (embeddings).
Muito mais preciso que histogramas/landmarks manuais.
"""
import cv2
import numpy as np
import os
import urllib.request
from typing import List, Tuple, Optional, Dict
from database import Database


MODEL_DIR = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'FaceID', 'models')
PROJECT_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
YUNET_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
SFACE_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'
YUNET_NAME = 'face_detection_yunet_2023mar.onnx'
SFACE_NAME = 'face_recognition_sface_2021dec.onnx'


def _ensure_model(filename: str, url: str) -> str:
    """
    Garante o modelo em um caminho sem acentos (OpenCV no Windows falha com 'Área de Trabalho').
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, filename)
    project_src = os.path.join(PROJECT_MODELS, filename)

    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest

    if os.path.exists(project_src) and os.path.getsize(project_src) > 1000:
        import shutil
        shutil.copy2(project_src, dest)
        return dest

    print(f'Baixando modelo: {filename}...')
    urllib.request.urlretrieve(url, dest)
    # Espelha no projeto se possível
    try:
        os.makedirs(PROJECT_MODELS, exist_ok=True)
        import shutil
        shutil.copy2(dest, project_src)
    except Exception:
        pass
    return dest


class FaceRecognizer:
    def __init__(self, database: Database, threshold: float = 0.363):
        """
        threshold: similaridade cosseno do SFace (oficial ~0.363).
        Valores maiores = mais rigoroso.
        """
        self.database = database
        self.threshold = threshold
        self.known_faces: Dict[int, List[np.ndarray]] = {}

        yunet_path = _ensure_model(YUNET_NAME, YUNET_URL)
        sface_path = _ensure_model(SFACE_NAME, SFACE_URL)

        self.detector = cv2.FaceDetectorYN.create(
            yunet_path, '', (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, '')
        self._input_size = (320, 320)

        # Fallback Haar se YuNet falhar
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.eye_glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )

        self.load_known_faces()

    def _set_detector_size(self, width: int, height: int):
        size = (int(width), int(height))
        if size != self._input_size:
            self.detector.setInputSize(size)
            self._input_size = size

    def detect_faces_detailed(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta rostos com YuNet.
        Retorna lista de dicts: location (x,y,w,h), landmarks, score, face_row
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        self._set_detector_size(w, h)
        _, faces = self.detector.detect(image)

        results = []
        if faces is not None:
            for face in faces:
                x, y, fw, fh = face[:4].astype(int)
                x = max(0, x)
                y = max(0, y)
                fw = max(1, min(fw, w - x))
                fh = max(1, min(fh, h - y))
                score = float(face[14]) if len(face) > 14 else float(face[-1])

                # Landmarks YuNet: right_eye, left_eye, nose, right_mouth, left_mouth
                landmarks = {
                    'right_eye': (float(face[4]), float(face[5])),
                    'left_eye': (float(face[6]), float(face[7])),
                    'nose': (float(face[8]), float(face[9])),
                    'right_mouth': (float(face[10]), float(face[11])),
                    'left_mouth': (float(face[12]), float(face[13])),
                    'mouth': (
                        (float(face[10]) + float(face[12])) / 2.0,
                        (float(face[11]) + float(face[13])) / 2.0,
                    ),
                    'eyes_detected': True,
                }

                results.append({
                    'location': (x, y, fw, fh),
                    'landmarks': landmarks,
                    'score': score,
                    'face_row': face.copy(),
                })

        if results:
            return results

        # Fallback Haar
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        haar = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, fw, fh) in haar:
            results.append({
                'location': (int(x), int(y), int(fw), int(fh)),
                'landmarks': None,
                'score': 0.5,
                'face_row': None,
            })
        return results

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [r['location'] for r in self.detect_faces_detailed(image)]

    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        height, width = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def _embed_from_face_row(self, image: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
        try:
            aligned = self.recognizer.alignCrop(image, face_row)
            feature = self.recognizer.feature(aligned)
            return np.asarray(feature, dtype=np.float32).flatten()
        except Exception as e:
            print(f'Erro ao extrair embedding SFace: {e}')
            return None

    def _embed_from_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Fallback: monta face_row aproximado a partir do bbox."""
        x, y, w, h = bbox
        # Formato YuNet: x,y,w,h + 5 landmarks (x,y) + score
        face_row = np.array([
            x, y, w, h,
            x + 0.3 * w, y + 0.35 * h,  # right eye (YuNet order)
            x + 0.7 * w, y + 0.35 * h,  # left eye
            x + 0.5 * w, y + 0.5 * h,   # nose
            x + 0.35 * w, y + 0.72 * h, # right mouth
            x + 0.65 * w, y + 0.72 * h, # left mouth
            1.0
        ], dtype=np.float32)
        return self._embed_from_face_row(image, face_row)

    def extract_embedding(self, image: np.ndarray, face_info: Optional[Dict] = None) -> Optional[np.ndarray]:
        if face_info is None:
            faces = self.detect_faces_detailed(image)
            if not faces:
                return None
            face_info = max(faces, key=lambda f: f['score'])

        if face_info.get('face_row') is not None:
            emb = self._embed_from_face_row(image, face_info['face_row'])
            if emb is not None:
                return emb

        return self._embed_from_bbox(image, face_info['location'])

    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        f1 = np.asarray(features1, dtype=np.float32).reshape(1, -1)
        f2 = np.asarray(features2, dtype=np.float32).reshape(1, -1)
        try:
            score = float(self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE))
        except Exception:
            n1 = np.linalg.norm(f1)
            n2 = np.linalg.norm(f2)
            if n1 == 0 or n2 == 0:
                return 0.0
            score = float(np.dot(f1.flatten(), f2.flatten()) / (n1 * n2))
        return float(np.clip(score, -1.0, 1.0))

    def load_known_faces(self):
        """Carrega embeddings do banco; se vazio, gera a partir das imagens."""
        self.known_faces = {}

        encodings = self.database.buscar_todos_encodings()
        for item in encodings:
            uid = int(item['usuario_id'])
            emb = np.asarray(item['encoding'], dtype=np.float32).flatten()
            self.known_faces.setdefault(uid, []).append(emb)

        # Migração: usuários com foto mas sem encoding no DB
        usuarios = self.database.listar_usuarios()
        for usuario in usuarios:
            uid = usuario['id']
            if uid in self.known_faces and self.known_faces[uid]:
                continue
            user_dir = os.path.join('faces', str(uid))
            if not os.path.isdir(user_dir):
                continue
            for filename in os.listdir(user_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                image = cv2.imread(os.path.join(user_dir, filename))
                if image is None:
                    continue
                emb = self.extract_embedding(image)
                if emb is None:
                    continue
                self.database.adicionar_encoding(uid, emb)
                self.known_faces.setdefault(uid, []).append(emb)

    def recognize_face(self, face_image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None,
                       face_info: Optional[Dict] = None) -> Optional[Dict]:
        if face_info is None and face_bbox is not None:
            face_info = {'location': face_bbox, 'face_row': None, 'score': 1.0}

        features = self.extract_embedding(face_image, face_info)
        if features is None:
            return None

        if not self.known_faces:
            return None

        best_match = None
        best_score = -1.0
        second_best = -1.0

        for usuario_id, embeddings in self.known_faces.items():
            # Usa a melhor similaridade entre as amostras do usuário
            scores = [self.compare_features(features, emb) for emb in embeddings]
            user_best = max(scores) if scores else -1.0
            if user_best > best_score:
                second_best = best_score
                best_score = user_best
                best_match = usuario_id
            elif user_best > second_best:
                second_best = user_best

        # Match válido: acima do threshold e com margem sobre o 2º lugar
        if best_match is None or best_score < self.threshold:
            return None

        margin = best_score - second_best
        if second_best > 0 and margin < 0.03 and best_score < self.threshold + 0.08:
            return None

        usuario = self.database.buscar_usuario(best_match)
        if not usuario:
            return None

        return {
            'usuario_id': int(best_match),
            'nome': str(usuario['nome']),
            'confidence': float(best_score),
            'distance': float(1.0 - best_score),
        }

    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        faces = self.detect_faces_detailed(image)
        results = []

        for face in faces:
            x, y, w, h = face['location']
            recognition = self.recognize_face(image, face['location'], face_info=face)

            if recognition:
                results.append({
                    'location': (int(x), int(y), int(w), int(h)),
                    'usuario_id': recognition['usuario_id'],
                    'nome': recognition['nome'],
                    'confidence': recognition['confidence'],
                })
            else:
                results.append({
                    'location': (int(x), int(y), int(w), int(h)),
                    'usuario_id': None,
                    'nome': 'Desconhecido',
                    'confidence': 0.0,
                })

        return results

    def validate_face_quality(self, face_image: np.ndarray, landmarks: Optional[Dict] = None) -> Tuple[bool, str]:
        if face_image is None or face_image.size == 0:
            return False, 'Imagem inválida'

        h, w = face_image.shape[:2]
        if w < 80 or h < 80:
            return False, 'Rosto muito pequeno. Aproxime-se da câmera.'

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 35:
            return False, 'Imagem muito escura. Melhore a iluminação.'
        if mean_brightness > 230:
            return False, 'Imagem muito clara. Reduza a iluminação.'

        return True, 'OK'

    def detect_landmarks_advanced(self, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """Compatibilidade com endpoints antigos."""
        faces = self.detect_faces_detailed(face_image)
        for face in faces:
            fx, fy, fw, fh = face['location']
            x, y, w, h = face_bbox
            # Interseção aproximada
            if abs(fx - x) < w * 0.5 and abs(fy - y) < h * 0.5:
                return face.get('landmarks')
        if faces:
            return faces[0].get('landmarks')
        return None

    def extract_features_from_array(self, face_image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        face_info = None
        if face_bbox is not None:
            face_info = {'location': face_bbox, 'face_row': None, 'score': 1.0}
        return self.extract_embedding(face_image, face_info)

    def add_face(self, usuario_id: int, face_image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None,
                 face_info: Optional[Dict] = None) -> bool:
        if face_info is None and face_bbox is not None:
            face_info = {'location': face_bbox, 'face_row': None, 'score': 1.0}
        elif face_info is None:
            faces = self.detect_faces_detailed(face_image)
            if not faces:
                return False
            face_info = max(faces, key=lambda f: f['score'])

        emb = self.extract_embedding(face_image, face_info)
        if emb is None:
            return False

        self.database.adicionar_encoding(usuario_id, emb)
        self.known_faces.setdefault(usuario_id, []).append(emb)
        return True

    def validate_and_add_face(self, usuario_id: int, image: np.ndarray,
                              face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, str]:
        faces = self.detect_faces_detailed(image)
        if not faces:
            return False, 'Nenhum rosto detectado'

        face = max(faces, key=lambda f: f['score'])
        if face_bbox is not None:
            # Prefere o bbox informado se houver match
            for f in faces:
                if f['location'] == face_bbox:
                    face = f
                    break

        region = self.extract_face_region(image, face['location'])
        is_valid, message = self.validate_face_quality(region, face.get('landmarks'))
        if not is_valid:
            return False, message

        if not self.add_face(usuario_id, image, face['location'], face_info=face):
            return False, 'Erro ao extrair características faciais'

        return True, message
