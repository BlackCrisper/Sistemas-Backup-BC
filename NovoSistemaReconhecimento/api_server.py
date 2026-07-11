"""
Microserviço Face ID para integração com RefeiControl.
Entrada de produção (sem YOLO/ultralytics) — YuNet + SFace + Cloudinary.
"""
import os
from functools import wraps

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, request, jsonify
from flask_cors import CORS

from database import Database
from face_recognizer_advanced import FaceRecognizer
from facial_analysis import FacialAnalysis
from face_enroll import (
    collect_images_from_payload,
    enroll_faces_for_user,
    recognize_1n,
)
import cloudinary_storage

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

FACE_API_KEY = os.environ.get('FACE_API_KEY', '').strip()
CANDIDATE_THRESHOLD = float(os.environ.get('FACE_MATCH_THRESHOLD', '0.70'))

db = Database()

# Restaura rostos/embeddings do Cloudinary (sem precisar de Disk pago no Render)
try:
    import cloudinary_db
    restored = cloudinary_db.load_state_from_cloudinary(db)
    if restored:
        print(f'Boot: {restored} usuário(s) faciais restaurados do Cloudinary.')
except Exception as e:
    print(f'Boot: não foi possível restaurar Cloudinary ({e})')

face_recognizer = FaceRecognizer(db)
facial_analysis = FacialAnalysis()


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not FACE_API_KEY:
            # Em desenvolvimento sem key, permite; em produção configure FACE_API_KEY
            if os.environ.get('NODE_ENV') == 'production' or os.environ.get('REQUIRE_FACE_API_KEY') == '1':
                return jsonify({'error': 'FACE_API_KEY não configurada no servidor', 'code': 'MISCONFIGURED'}), 500
            return f(*args, **kwargs)
        key = request.headers.get('X-Face-API-Key', '')
        if key != FACE_API_KEY:
            return jsonify({'error': 'API key inválida', 'code': 'UNAUTHORIZED'}), 401
        return f(*args, **kwargs)
    return decorated


def _json_or_form():
    if request.is_json:
        return request.get_json(silent=True) or {}
    data = request.form.to_dict() if request.form else {}
    return data


def _decode_image_from_request():
    from face_enroll import decode_base64_image, collect_images_from_payload
    data = _json_or_form()
    images = collect_images_from_payload(data, request.files)
    return images[0] if images else None


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'face-id',
        'users': len(db.listar_usuarios()),
        'cloudinary': cloudinary_storage.is_configured(),
        'persistence': 'cloudinary' if cloudinary_storage.is_configured() else 'ephemeral',
        'api_key_required': bool(FACE_API_KEY),
        'match_threshold': CANDIDATE_THRESHOLD,
    })


@app.route('/api/detect', methods=['POST'])
@require_api_key
def api_detect():
    """Detecta rosto + checklist de qualidade para captura automática."""
    import cv2
    import numpy as np
    from face_enroll import decode_base64_image, collect_images_from_payload

    data = _json_or_form()
    images = collect_images_from_payload(data, request.files)
    if not images:
        return jsonify({'error': 'Nenhuma imagem fornecida'}), 400

    image = images[0]
    h, w = image.shape[:2]
    max_side = 420
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    detailed = face_recognizer.detect_faces_detailed(image)
    results = []
    for face in detailed:
        x, y, fw, fh = face['location']
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            fw = int(fw / scale)
            fh = int(fh / scale)

        landmarks = face.get('landmarks')
        scaled_landmarks = None
        if landmarks and scale != 1.0:
            scaled_landmarks = {}
            for key, value in landmarks.items():
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    scaled_landmarks[key] = [float(value[0]) / scale, float(value[1]) / scale]
                else:
                    scaled_landmarks[key] = value
        else:
            scaled_landmarks = landmarks

        fx, fy, ffw, ffh = face['location']
        try:
            facial_info = facial_analysis.analyze_full_face(
                image, (fx, fy, ffw, ffh),
                landmarks=face.get('landmarks'),
                frame_shape=image.shape,
            )
        except Exception as e:
            print(f'Erro análise facial: {e}')
            facial_info = {
                'capture_ready': False,
                'capture_blockers': ['falha na análise'],
                'checks': [],
                'scan_progress': 0.0,
            }

        results.append({
            'location': [int(x), int(y), int(fw), int(fh)],
            'landmarks': {
                k: ([float(v[0]), float(v[1])] if isinstance(v, (tuple, list)) and len(v) == 2 else v)
                for k, v in (scaled_landmarks or {}).items()
            } if scaled_landmarks else None,
            'facial_analysis': {
                'glasses': bool(facial_info.get('glasses', False)),
                'sunglasses': bool(facial_info.get('sunglasses', False)),
                'hat': bool(facial_info.get('hat', False)),
                'eyes_open': bool(facial_info.get('eyes_open', False)),
                'left_eye': str(facial_info.get('left_eye', 'unknown')),
                'right_eye': str(facial_info.get('right_eye', 'unknown')),
                'capture_ready': bool(facial_info.get('capture_ready', False)),
                'capture_blockers': facial_info.get('capture_blockers', []),
                'quality': str(facial_info.get('quality', 'poor')),
                'brightness': float(facial_info.get('brightness', 0.0)),
                'sharpness': float(facial_info.get('sharpness', 0.0)),
                'pose_ok': bool(facial_info.get('pose_ok', False)),
                'position_ok': bool(facial_info.get('position_ok', False)),
                'scan_progress': float(facial_info.get('scan_progress', 0.0)),
                'checks': facial_info.get('checks', []),
            },
        })

    return jsonify({'success': True, 'results': results})


@app.route('/api/enroll', methods=['POST'])
@require_api_key
def api_enroll():
    """
    Cadastra ou atualiza rosto por matrícula.
    Body JSON: { matricula, nome, images: [base64...], replace?: true }
    """
    data = _json_or_form()
    nome = (data.get('nome') or '').strip()
    matricula = Database.normalizar_matricula(data.get('matricula', ''))
    replace = str(data.get('replace', 'true')).lower() in ('1', 'true', 'yes')

    if not nome:
        return jsonify({'error': 'Nome é obrigatório', 'code': 'NAME_REQUIRED'}), 400
    if not matricula:
        return jsonify({'error': 'Matrícula deve ter exatamente 3 dígitos', 'code': 'INVALID_MATRICULA'}), 400

    images = collect_images_from_payload(data, request.files)
    if not images:
        return jsonify({'error': 'Nenhuma imagem fornecida', 'code': 'NO_IMAGE'}), 400

    usuario = db.buscar_usuario_por_matricula(matricula)
    if usuario:
        usuario_id = int(usuario['id'])
        db.atualizar_usuario(usuario_id, nome, matricula)
    else:
        usuario_id = db.criar_usuario(nome, matricula)
        if usuario_id is None:
            return jsonify({'error': 'Erro ao criar usuário', 'code': 'CREATE_FAILED'}), 400

    added, last_error = enroll_faces_for_user(
        db, face_recognizer, facial_analysis,
        usuario_id, images, matricula, replace=replace, nome=nome,
    )
    if added == 0:
        return jsonify({'error': last_error, 'code': 'ENROLL_FAILED'}), 400

    photos = db.listar_fotos_cloudinary(usuario_id)
    return jsonify({
        'success': True,
        'message': f'Rosto cadastrado com {added} amostra(s).',
        'usuario_id': usuario_id,
        'matricula': matricula,
        'nome': nome,
        'samples': added,
        'photos': [
            {'url': p.get('cloudinary_url'), 'public_id': p.get('cloudinary_public_id')}
            for p in photos
        ],
        'enrolled': True,
    })


@app.route('/api/simulacao/reconhecer', methods=['POST'])
@app.route('/api/recognize', methods=['POST'])
@require_api_key
def api_recognize():
    """Reconhecimento 1:N (threshold de negócio configurável)."""
    data = _json_or_form()
    images = collect_images_from_payload(data, request.files)
    if not images:
        return jsonify({'success': False, 'matched': False, 'error': 'Nenhuma imagem fornecida'}), 400

    result = recognize_1n(face_recognizer, images[0], threshold=CANDIDATE_THRESHOLD)
    if result.get('matched') and result.get('usuario_id'):
        usuario = db.buscar_usuario(int(result['usuario_id']))
        if usuario:
            result['nome'] = usuario['nome']
            result['matricula'] = usuario.get('matricula')
            photos = db.listar_fotos_cloudinary(usuario['id'])
            result['foto_url'] = photos[0]['cloudinary_url'] if photos else None
        else:
            result['matched'] = False
            result['message'] = 'Usuário não encontrado'
    return jsonify(result)


@app.route('/api/users/by-matricula/<matricula>', methods=['GET', 'DELETE'])
@require_api_key
def api_user_by_matricula(matricula):
    matricula = Database.normalizar_matricula(matricula)
    if not matricula:
        return jsonify({'error': 'Matrícula inválida', 'code': 'INVALID_MATRICULA'}), 400

    usuario = db.buscar_usuario_por_matricula(matricula)
    if not usuario:
        return jsonify({'error': 'Usuário não encontrado', 'code': 'NOT_FOUND'}), 404

    if request.method == 'GET':
        photos = db.listar_fotos_cloudinary(usuario['id'])
        return jsonify({
            'success': True,
            'usuario': {
                'id': usuario['id'],
                'nome': usuario['nome'],
                'matricula': usuario.get('matricula'),
                'enrolled': db.tem_rosto_cadastrado(usuario['id']),
                'foto_url': photos[0]['cloudinary_url'] if photos else None,
                'photos': [p.get('cloudinary_url') for p in photos],
            },
        })

    # DELETE
    cloudinary_storage.delete_folder_for_matricula(matricula, usuario.get('nome') or '')
    uid = int(usuario['id'])
    face_recognizer.known_faces.pop(uid, None)
    db.deletar_usuario(uid)
    try:
        import cloudinary_db
        cloudinary_db.sync_after_change(db, face_recognizer)
    except Exception as e:
        print(f'Falha ao sincronizar Cloudinary após delete: {e}')
    return jsonify({'success': True, 'message': 'Usuário facial removido', 'matricula': matricula})


@app.route('/api/users/enrolled', methods=['GET'])
@require_api_key
def api_users_enrolled():
    """Lista matrículas com rosto cadastrado (para a tela de usuários do RefeiControl)."""
    enrolled = []
    for u in db.listar_usuarios():
        uid = int(u['id'])
        if not db.tem_rosto_cadastrado(uid):
            continue
        mat = u.get('matricula')
        if not mat:
            continue
        enrolled.append({
            'matricula': mat,
            'nome': u.get('nome'),
            'enrolled': True,
        })
    return jsonify({
        'success': True,
        'count': len(enrolled),
        'matriculas': [e['matricula'] for e in enrolled],
        'users': enrolled,
    })


@app.route('/api/matricula/<matricula>', methods=['GET'])
@require_api_key
def api_lookup_matricula(matricula):
    matricula = Database.normalizar_matricula(matricula)
    if not matricula:
        return jsonify({'error': 'Matrícula inválida'}), 400
    usuario = db.buscar_usuario_por_matricula(matricula)
    if not usuario:
        return jsonify({'error': 'Não encontrado'}), 404
    photos = db.listar_fotos_cloudinary(usuario['id'])
    return jsonify({
        'id': usuario['id'],
        'nome': usuario['nome'],
        'matricula': usuario.get('matricula'),
        'enrolled': db.tem_rosto_cadastrado(usuario['id']),
        'foto_url': photos[0]['cloudinary_url'] if photos else None,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5001'))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG') == '1')
