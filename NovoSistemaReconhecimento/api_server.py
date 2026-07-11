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
CANDIDATE_THRESHOLD = float(os.environ.get('FACE_MATCH_THRESHOLD', '0.80'))

db = Database()
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
    # Também aceita campos JSON misturados
    return data


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'face-id',
        'users': len(db.listar_usuarios()),
        'cloudinary': cloudinary_storage.is_configured(),
        'api_key_required': bool(FACE_API_KEY),
    })


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
        usuario_id, images, matricula, replace=replace,
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
    """Reconhecimento 1:N (threshold de negócio padrão 0.80)."""
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
    cloudinary_storage.delete_folder_for_matricula(matricula)
    uid = int(usuario['id'])
    face_recognizer.known_faces.pop(uid, None)
    db.deletar_usuario(uid)
    return jsonify({'success': True, 'message': 'Usuário facial removido', 'matricula': matricula})


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
