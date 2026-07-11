"""
Lógica compartilhada de enroll / recognize para o microserviço Face ID.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

import cloudinary_storage


def decode_base64_image(base64_string):
    """Decodifica imagem base64 para array OpenCV BGR."""
    import base64
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return b''
    return buf.tobytes()


def enroll_faces_for_user(
    db,
    face_recognizer,
    facial_analysis,
    usuario_id: int,
    images: List[np.ndarray],
    matricula: str,
    replace: bool = False,
) -> Tuple[int, str]:
    """
    Valida e cadastra faces. Uploads vão para Cloudinary quando configurado.
    Retorna (added_count, last_error).
    """
    if replace:
        photos = db.listar_fotos_cloudinary(usuario_id)
        cloudinary_storage.delete_folder_for_matricula(matricula)
        db.limpar_encodings(usuario_id)
        db.limpar_fotos(usuario_id)
        face_recognizer.known_faces.pop(int(usuario_id), None)
        del photos  # noqa: F841

    added = 0
    last_error = 'Nenhum rosto válido encontrado'
    photo_index = 0

    for image in images:
        detailed = face_recognizer.detect_faces_detailed(image)
        if not detailed:
            last_error = 'Nenhum rosto detectado. Centralize o rosto e melhore a iluminação.'
            continue
        face, has_multiple = face_recognizer.select_primary_face(detailed)
        if has_multiple:
            last_error = 'Múltiplos rostos detectados. Cadastre uma pessoa por vez.'
            continue
        if face is None:
            last_error = 'Nenhum rosto detectado. Centralize o rosto e melhore a iluminação.'
            continue

        try:
            facial_info = facial_analysis.analyze_full_face(
                image, face['location'], landmarks=face.get('landmarks')
            )
        except Exception as e:
            print(f'Erro análise no cadastro: {e}')
            facial_info = {'capture_blockers': ['falha na análise'], 'capture_ready': False}

        if facial_info.get('sunglasses'):
            last_error = 'Remova os óculos escuros para cadastrar.'
            continue
        if facial_info.get('hat'):
            last_error = 'Remova chapéu/boné para cadastrar.'
            continue
        if facial_info.get('capture_blockers'):
            last_error = 'Corrija: ' + ', '.join(facial_info['capture_blockers'])
            continue

        success, message = face_recognizer.validate_and_add_face(
            usuario_id, image, face['location']
        )
        if not success:
            last_error = message
            continue

        jpeg = image_to_jpeg_bytes(image)
        uploaded = cloudinary_storage.upload_face_image(jpeg, matricula, photo_index) if jpeg else None
        if uploaded:
            public_id, url = uploaded
            db.adicionar_foto_cloudinary(usuario_id, public_id, url)
        photo_index += 1
        added += 1

    if added > 0:
        face_recognizer.load_known_faces()
        try:
            import cloudinary_db
            cloudinary_db.sync_after_change(db, face_recognizer)
        except Exception as e:
            print(f'Falha ao sincronizar Cloudinary após enroll: {e}')

    return added, last_error


def resize_for_inference(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / float(max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def recognize_1n(face_recognizer, image: np.ndarray, threshold: float = 0.70):
    """
    Reconhecimento 1:N.
    Só retorna matched=True com confiança alta o bastante — evita falso positivo
    quando há poucos rostos cadastrados.
    """
    import cv2
    import os

    n_users = len(face_recognizer.known_faces)
    enrolled = sum(len(v) for v in face_recognizer.known_faces.values())

    # Com poucos cadastros, exige score mais alto (1 rosto no banco tende a "aceitar" qualquer um)
    base = float(threshold)
    if n_users <= 1:
        effective_threshold = max(base, 0.72)
    elif n_users <= 5:
        effective_threshold = max(base, 0.65)
    else:
        effective_threshold = base

    # Override opcional via env
    min_env = os.environ.get('FACE_MATCH_THRESHOLD_MIN')
    if min_env:
        try:
            effective_threshold = max(effective_threshold, float(min_env))
        except ValueError:
            pass

    image = resize_for_inference(image)
    results = face_recognizer.detect_and_recognize(image)

    if not results:
        flipped = cv2.flip(image, 1)
        results = face_recognizer.detect_and_recognize(flipped)

    if not results:
        return {
            'success': True,
            'matched': False,
            'face_detected': False,
            'message': 'Nenhum rosto detectado — centralize o rosto e melhore a luz',
            'confidence': 0.0,
            'enrolled_embeddings': enrolled,
            'enrolled_users': n_users,
            'threshold_used': effective_threshold,
        }

    # Melhor candidato por score (mesmo sem match interno)
    best = max(results, key=lambda r: r.get('confidence') or 0.0)
    confidence = float(best.get('confidence') or 0.0)
    usuario_id = best.get('usuario_id') if best.get('usuario_id') else None

    # Reavalia match só pelo limiar de negócio (não confiar só no SFace 0.40)
    is_match = bool(usuario_id) and confidence >= effective_threshold

    if enrolled == 0 or n_users == 0:
        return {
            'success': True,
            'matched': False,
            'face_detected': True,
            'message': 'Rosto visto, mas não há ninguém com foto cadastrada. Cadastre no admin.',
            'confidence': confidence,
            'enrolled_embeddings': 0,
            'enrolled_users': 0,
            'threshold_used': effective_threshold,
        }

    if not is_match:
        return {
            'success': True,
            'matched': False,
            'face_detected': True,
            'message': (
                'Rosto não cadastrado neste sistema'
                if confidence < 0.45
                else f'Não identificado (confiança {int(confidence * 100)}%, mínimo {int(effective_threshold * 100)}%)'
            ),
            'confidence': confidence,
            'enrolled_embeddings': enrolled,
            'enrolled_users': n_users,
            'threshold_used': effective_threshold,
        }

    return {
        'success': True,
        'matched': True,
        'face_detected': True,
        'message': 'Identidade confirmada',
        'confidence': confidence,
        'usuario_id': int(usuario_id),
        'nome': best.get('nome'),
        'enrolled_embeddings': enrolled,
        'enrolled_users': n_users,
        'threshold_used': effective_threshold,
    }


def collect_images_from_payload(data: dict, files=None) -> List[np.ndarray]:
    """Extrai imagens de JSON (imagem_base64 / images[]) ou multipart."""
    images: List[np.ndarray] = []

    if data:
        if data.get('imagem_base64'):
            img = decode_base64_image(data['imagem_base64'])
            if img is not None:
                images.append(img)
        for key in sorted(data.keys()):
            if str(key).startswith('imagem_base64_'):
                img = decode_base64_image(data[key])
                if img is not None:
                    images.append(img)
        raw_list = data.get('images') or data.get('imagens') or []
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, str):
                    img = decode_base64_image(item)
                    if img is not None:
                        images.append(img)

    if files:
        file_list = files.getlist('imagem') if hasattr(files, 'getlist') else []
        if not file_list and files.get('imagem'):
            file_list = [files.get('imagem')]
        for file in file_list:
            if file and getattr(file, 'filename', None):
                nparr = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)

    return images
