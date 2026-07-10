"""
Aplicação Flask para sistema de reconhecimento facial
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import io

from database import Database
from face_detector import FaceDetector
from face_recognizer_advanced import FaceRecognizer
from liveness_detector import LivenessDetector
from facial_analysis import FacialAnalysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Inicializa componentes
db = Database()
face_detector = FaceDetector()
face_recognizer = FaceRecognizer(db)
liveness_detector = LivenessDetector()
facial_analysis = FacialAnalysis()

# Garante que os diretórios existam
os.makedirs('faces', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)


def allowed_file(filename):
    """Verifica se o arquivo tem extensão permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_image_from_base64(base64_string, save_path):
    """Salva uma imagem a partir de uma string base64"""
    try:
        # Remove o prefixo data:image se existir
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Converte para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Erro ao salvar imagem: {e}")
        return False


@app.route('/')
def index():
    """Página inicial"""
    usuarios = db.listar_usuarios()
    return render_template('index.html', usuarios=usuarios)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Página de cadastro de nova face"""
    if request.method == 'POST':
        nome = request.form.get('nome', '').strip()
        cpf = request.form.get('cpf', '').strip() or None

        if not nome:
            return jsonify({'error': 'Nome é obrigatório'}), 400

        # Aceita uma ou várias imagens (imagem_base64 / imagem_base64_0..)
        images = []

        # Upload de arquivo único
        if 'imagem' in request.files:
            file = request.files['imagem']
            if file and file.filename and allowed_file(file.filename):
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)

        # Base64 único
        if 'imagem_base64' in request.form and request.form['imagem_base64']:
            img = decode_base64_image(request.form['imagem_base64'])
            if img is not None:
                images.append(img)

        # Múltiplas capturas
        for key in sorted(request.form.keys()):
            if key.startswith('imagem_base64_'):
                img = decode_base64_image(request.form[key])
                if img is not None:
                    images.append(img)

        if not images:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400

        usuario_id = db.criar_usuario(nome, cpf)
        if usuario_id is None:
            return jsonify({'error': 'Erro ao criar usuário. CPF pode já estar cadastrado.'}), 400

        user_dir = os.path.join('faces', str(usuario_id))
        os.makedirs(user_dir, exist_ok=True)

        added = 0
        last_error = 'Nenhum rosto válido encontrado'

        for idx, image in enumerate(images):
            filepath = os.path.join(user_dir, f'face_{usuario_id}_{idx}.jpg')
            cv2.imwrite(filepath, image)

            faces = face_recognizer.detect_faces(image)
            if not faces:
                last_error = 'Nenhum rosto detectado. Centralize o rosto e melhore a iluminação.'
                continue
            if len(faces) > 1:
                last_error = 'Múltiplos rostos detectados. Cadastre uma pessoa por vez.'
                continue

            success, message = face_recognizer.validate_and_add_face(usuario_id, image, faces[0])
            if success:
                added += 1
            else:
                last_error = message

        if added == 0:
            db.deletar_usuario(usuario_id)
            return jsonify({'error': last_error}), 400

        face_recognizer.load_known_faces()

        return jsonify({
            'success': True,
            'message': f'Usuário cadastrado com {added} amostra(s).',
            'usuario_id': usuario_id,
            'samples': added,
        })

    return render_template('register.html')


def decode_base64_image(base64_string):
    """Decodifica imagem base64 para array OpenCV."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    """Página de reconhecimento facial"""
    if request.method == 'POST':
        if 'imagem' not in request.files and 'imagem_base64' not in request.form:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400

        image = None
        stream_mode = request.form.get('stream', '0') == '1'

        if 'imagem' in request.files:
            file = request.files['imagem']
            if file and file.filename and allowed_file(file.filename):
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None and 'imagem_base64' in request.form:
            image = decode_base64_image(request.form['imagem_base64'])

        if image is None:
            return jsonify({'error': 'Erro ao processar imagem'}), 400

        # Reduz resolução no stream para ganhar velocidade
        if stream_mode:
            h, w = image.shape[:2]
            max_side = 480
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                image = cv2.resize(image, (int(w * scale), int(h * scale)))

        results = face_recognizer.detect_and_recognize(image)

        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            return obj

        results_serializable = convert_to_native(results)
        response = {
            'success': True,
            'results': results_serializable,
        }

        # Imagem anotada só no modo upload (não no stream)
        if not stream_mode:
            labels = []
            for result in results:
                if result['usuario_id']:
                    labels.append(f"{result['nome']} ({result['confidence']:.2f})")
                else:
                    labels.append('Desconhecido')
            faces_locations = [r['location'] for r in results]
            annotated_image = face_detector.draw_faces(image, faces_locations, labels)
            _, buffer = cv2.imencode('.jpg', annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            response['image'] = f'data:image/jpeg;base64,{image_base64}'

        return jsonify(response)

    return render_template('recognize.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Detecta rostos + análise (olhos, óculos, chapéu) para o guia de cadastro."""
    image = None

    if 'imagem_base64' in request.form:
        image = decode_base64_image(request.form['imagem_base64'])
    elif 'imagem' in request.files:
        file = request.files['imagem']
        if file and file.filename:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Nenhuma imagem fornecida ou erro ao processar'}), 400

    # Reduz para acelerar o guia no mobile
    h, w = image.shape[:2]
    max_side = 420
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    detailed = face_recognizer.detect_faces_detailed(image)
    if not detailed:
        yolo_faces = face_detector.detect_faces(image)
        detailed = [{'location': f, 'landmarks': None, 'score': 0.5, 'face_row': None} for f in yolo_faces]

    results = []
    for face in detailed:
        x, y, w, h = face['location']
        # Reescala para coordenadas da imagem original enviada
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        landmarks = face.get('landmarks')
        # Landmarks também precisam de reescala
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

        # Análise usa a imagem já redimensionada + bbox nela
        fx, fy, fw, fh = face['location']
        try:
            facial_info = facial_analysis.analyze_full_face(
                image, (fx, fy, fw, fh), landmarks=face.get('landmarks')
            )
        except Exception as e:
            print(f'Erro análise facial: {e}')
            facial_info = {
                'glasses': False,
                'hat': False,
                'eyes_open': False,
                'left_eye': 'unknown',
                'right_eye': 'unknown',
                'accessories': [],
                'capture_ready': False,
                'capture_blockers': ['falha na análise'],
            }

        results.append({
            'location': [int(x), int(y), int(w), int(h)],
            'landmarks': {
                k: ([float(v[0]), float(v[1])] if isinstance(v, (tuple, list)) and len(v) == 2 else v)
                for k, v in (scaled_landmarks or {}).items()
            } if scaled_landmarks else None,
            'facial_analysis': {
                'glasses': bool(facial_info.get('glasses', False)),
                'glasses_confidence': float(facial_info.get('glasses_confidence', 0.0)),
                'hat': bool(facial_info.get('hat', False)),
                'hat_confidence': float(facial_info.get('hat_confidence', 0.0)),
                'left_eye': str(facial_info.get('left_eye', 'unknown')),
                'right_eye': str(facial_info.get('right_eye', 'unknown')),
                'eyes_open': bool(facial_info.get('eyes_open', False)),
                'accessories': facial_info.get('accessories', []),
                'capture_ready': bool(facial_info.get('capture_ready', False)),
                'capture_blockers': facial_info.get('capture_blockers', []),
                'quality': str(facial_info.get('quality', 'poor')),
            }
        })

    return jsonify({
        'success': True,
        'results': results
    })


@app.route('/api/detect-detailed', methods=['POST'])
def api_detect_detailed():
    """API REST para detectar rostos com landmarks e informações detalhadas"""
    # Aceita tanto base64 quanto arquivo
    image = None
    
    if 'imagem_base64' in request.form:
        base64_string = request.form['imagem_base64']
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        try:
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Erro ao decodificar imagem: {str(e)}'}), 400
    elif 'imagem' in request.files:
        file = request.files['imagem']
        if file and file.filename:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Nenhuma imagem fornecida ou erro ao processar'}), 400
    
    # Detecta rostos
    faces = face_recognizer.detect_faces(image)
    
    # Se não detectou com OpenCV, tenta YOLO
    if not faces:
        faces = face_detector.detect_faces(image)
    
    # Para cada rosto, extrai landmarks e informações
    results = []
    try:
        for face in faces:
            x, y, w, h = face
            
            # Valida bounding box
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue
            
            # Analisa características faciais (com tratamento de erro)
            facial_info = None
            try:
                facial_info = facial_analysis.analyze_full_face(image, (x, y, w, h))
            except Exception as e:
                print(f"Erro ao analisar características faciais: {e}")
                # Cria informações padrão em caso de erro
                facial_info = {
                    'glasses': False,
                    'glasses_confidence': 0.0,
                    'left_eye': 'unknown',
                    'right_eye': 'unknown',
                    'left_eye_confidence': 0.0,
                    'right_eye_confidence': 0.0,
                    'eyes_detected': 0
                }
            
            # Detecta landmarks usando o método correto
            landmarks = None
            try:
                landmarks = face_recognizer.detect_landmarks_advanced(image, (x, y, w, h))
            except Exception as e:
                print(f"Erro ao detectar landmarks: {e}")
                landmarks = None
            
            # Os landmarks já vêm com coordenadas absolutas da imagem completa
            adjusted_landmarks = {}
            if landmarks:
                for key, value in landmarks.items():
                    if key == 'eyes_detected':
                        adjusted_landmarks[key] = bool(value)
                    elif isinstance(value, (tuple, list)) and len(value) == 2:
                        # Converte para lista de inteiros (já são coordenadas absolutas)
                        try:
                            adjusted_landmarks[key] = [int(value[0]), int(value[1])]
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"Erro ao converter landmark {key}: {e}")
                            continue
            
            # Extrai características para verificar qualidade
            features = None
            feature_info = None
            try:
                features = face_recognizer.extract_features_from_array(image, (x, y, w, h))
                if features is not None:
                    feature_info = {
                        'feature_vector_size': len(features),
                        'has_features': True
                    }
            except Exception as e:
                print(f"Erro ao extrair características: {e}")
                feature_info = None
            
            # Analisa características faciais (com tratamento de erro)
            facial_info = None
            try:
                facial_info = facial_analysis.analyze_full_face(image, (x, y, w, h))
            except Exception as e:
                print(f"Erro ao analisar características faciais: {e}")
                import traceback
                traceback.print_exc()
                # Cria informações padrão em caso de erro
                facial_info = {
                    'glasses': False,
                    'glasses_confidence': 0.0,
                    'left_eye': 'unknown',
                    'right_eye': 'unknown',
                    'left_eye_confidence': 0.0,
                    'right_eye_confidence': 0.0,
                    'eyes_detected': 0
                }
            
            result_dict = {
                'location': [int(x), int(y), int(w), int(h)],
                'landmarks': adjusted_landmarks,
                'features': feature_info
            }
            
            # Adiciona análise facial apenas se disponível
            if facial_info:
                try:
                    result_dict['facial_analysis'] = {
                        'glasses': bool(facial_info.get('glasses', False)),
                        'glasses_confidence': float(facial_info.get('glasses_confidence', 0.0)),
                        'left_eye': str(facial_info.get('left_eye', 'unknown')),
                        'right_eye': str(facial_info.get('right_eye', 'unknown')),
                        'left_eye_confidence': float(facial_info.get('left_eye_confidence', 0.0)),
                        'right_eye_confidence': float(facial_info.get('right_eye_confidence', 0.0)),
                        'eyes_detected': int(facial_info.get('eyes_detected', 0))
                    }
                except Exception as e:
                    print(f"Erro ao adicionar análise facial ao resultado: {e}")
            
            results.append(result_dict)
    except Exception as e:
        return jsonify({'error': f'Erro ao processar rostos: {str(e)}'}), 500
    
    # Desenha imagem anotada
    try:
        annotated_image = image.copy()
        for result in results:
            x, y, w, h = result['location']
            # Desenha bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Desenha landmarks
            landmarks = result.get('landmarks', {})
            if landmarks:
                # Olhos
                if 'left_eye' in landmarks and isinstance(landmarks['left_eye'], list) and len(landmarks['left_eye']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['left_eye']), 5, (255, 0, 0), -1)
                if 'right_eye' in landmarks and isinstance(landmarks['right_eye'], list) and len(landmarks['right_eye']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['right_eye']), 5, (255, 0, 0), -1)
                # Nariz
                if 'nose' in landmarks and isinstance(landmarks['nose'], list) and len(landmarks['nose']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['nose']), 5, (0, 255, 255), -1)
                # Boca
                if 'mouth' in landmarks and isinstance(landmarks['mouth'], list) and len(landmarks['mouth']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['mouth']), 5, (255, 0, 255), -1)
                # Queixo
                if 'chin' in landmarks and isinstance(landmarks['chin'], list) and len(landmarks['chin']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['chin']), 5, (0, 255, 255), -1)
                # Testa
                if 'forehead' in landmarks and isinstance(landmarks['forehead'], list) and len(landmarks['forehead']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['forehead']), 5, (255, 255, 0), -1)
                # Bochechas
                if 'left_cheek' in landmarks and isinstance(landmarks['left_cheek'], list) and len(landmarks['left_cheek']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['left_cheek']), 5, (0, 255, 255), -1)
                if 'right_cheek' in landmarks and isinstance(landmarks['right_cheek'], list) and len(landmarks['right_cheek']) == 2:
                    cv2.circle(annotated_image, tuple(landmarks['right_cheek']), 5, (0, 255, 255), -1)
        
        # Converte imagem anotada para base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'results': results,
            'annotated_image': f'data:image/jpeg;base64,{image_base64}'
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Erro ao processar imagem anotada: {error_trace}")
        return jsonify({
            'success': True,
            'results': results,
            'annotated_image': None,
            'error': f'Erro ao gerar imagem anotada: {str(e)}'
        })


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API REST para reconhecimento facial"""
    # Aceita tanto base64 quanto arquivo
    image = None
    
    if 'imagem_base64' in request.form:
        base64_string = request.form['imagem_base64']
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        try:
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Erro ao decodificar imagem: {str(e)}'}), 400
    elif 'imagem' in request.files:
        file = request.files['imagem']
        if file and file.filename:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Nenhuma imagem fornecida ou erro ao processar'}), 400
    
    # Liveness apenas informativo (não bloqueia o reconhecimento via API)
    is_real, liveness_message, liveness_confidence = liveness_detector.validate_liveness(image, require_movement=False)
    
    # Detecta e reconhece faces
    results = face_recognizer.detect_and_recognize(image)
    
    # Adiciona análise facial para cada rosto detectado
    for result in results:
        if 'location' in result:
            face_bbox = result['location']
            facial_info = facial_analysis.analyze_full_face(image, face_bbox)
            result['facial_analysis'] = {
                'glasses': facial_info['glasses'],
                'glasses_confidence': facial_info['glasses_confidence'],
                'left_eye': facial_info['left_eye'],
                'right_eye': facial_info['right_eye'],
                'left_eye_confidence': facial_info['left_eye_confidence'],
                'right_eye_confidence': facial_info['right_eye_confidence']
            }
    
    # Converte valores NumPy para tipos nativos do Python
    def convert_to_native(obj):
        """Converte valores NumPy para tipos nativos do Python"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results_serializable = convert_to_native(results)
    
    return jsonify({
        'success': True,
        'faces_detected': len(results),
        'results': results_serializable,
        'liveness': {
            'is_real': bool(is_real),
            'message': liveness_message,
            'confidence': float(liveness_confidence)
        }
    })


@app.route('/api/users', methods=['GET'])
def api_users():
    """API para listar usuários"""
    usuarios = db.listar_usuarios()
    return jsonify({'usuarios': usuarios})


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    """API para deletar usuário"""
    db.deletar_usuario(user_id)
    return jsonify({'success': True, 'message': 'Usuário deletado com sucesso'})


if __name__ == '__main__':
    # Escuta em todas as interfaces + HTTPS (necessário para câmera no celular)
    import socket
    from datetime import datetime, timedelta, timezone

    def find_free_port(start_port=5000, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                continue
        return None

    def get_lan_ip():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                return s.getsockname()[0]
        except OSError:
            return '127.0.0.1'

    def ensure_ssl_certs(lan_ip):
        """Gera certificado autoassinado com IP da rede (para câmera no celular)."""
        cert_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certs')
        os.makedirs(cert_dir, exist_ok=True)
        cert_file = os.path.join(cert_dir, 'cert.pem')
        key_file = os.path.join(cert_dir, 'key.pem')
        meta_file = os.path.join(cert_dir, 'ip.txt')

        # Reutiliza se já existe para o mesmo IP
        if os.path.exists(cert_file) and os.path.exists(key_file) and os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                if f.read().strip() == lan_ip:
                    return cert_file, key_file

        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            print("Instale: pip install cryptography")
            print("Usando certificado ad-hoc (pode falhar no celular).")
            return 'adhoc'

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'FaceID Local'),
        ])
        alt_names = [
            x509.DNSName('localhost'),
            x509.IPAddress(__import__('ipaddress').IPv4Address('127.0.0.1')),
        ]
        try:
            alt_names.append(x509.IPAddress(__import__('ipaddress').IPv4Address(lan_ip)))
        except ValueError:
            pass

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
            .sign(key, hashes.SHA256())
        )

        with open(key_file, 'wb') as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        with open(cert_file, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write(lan_ip)

        return cert_file, key_file

    port = find_free_port()
    if port is None:
        print("Erro: Não foi possível encontrar uma porta disponível")
        exit(1)

    lan_ip = get_lan_ip()
    ssl_context = ensure_ssl_certs(lan_ip)

    print(f"\n{'='*50}")
    print("Servidor Flask com HTTPS iniciado!")
    print(f"Neste PC:     https://localhost:{port}")
    print(f"No celular:   https://{lan_ip}:{port}")
    print("")
    print("No celular: aceite o aviso de certificado")
    print("(Avançado > Continuar / Prosseguir mesmo assim)")
    print("Depois permita o acesso à câmera.")
    print(f"{'='*50}\n")

    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=port,
            use_reloader=False,
            ssl_context=ssl_context,
        )
    except OSError as e:
        print(f"Erro ao iniciar servidor: {e}")
        print("Tente fechar outros programas que possam estar usando a porta")
        print("Se o Firewall do Windows pedir, permita acesso na rede privada.")
