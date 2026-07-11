"""
Persistência gratuita do banco Face ID via Cloudinary (sem Disk no Render).

Salva um JSON com usuários + embeddings + metadados de fotos.
No boot do serviço, restaura o SQLite local (/tmp) a partir desse arquivo.
"""
import base64
import io
import json
import pickle
import time
from typing import Any, Dict

import cloudinary_storage

STATE_PUBLIC_ID = 'refeicontrol/face-db/state'


def export_state(db) -> Dict[str, Any]:
    """Monta snapshot serializável do banco."""
    usuarios = db.listar_usuarios()
    encodings = []
    photos = []

    for u in usuarios:
        uid = int(u['id'])
        for emb in db.buscar_encodings_usuario(uid):
            encodings.append({
                'usuario_id': uid,
                'encoding_b64': base64.b64encode(pickle.dumps(emb)).decode('ascii'),
            })
        for p in db.listar_fotos_cloudinary(uid):
            photos.append({
                'usuario_id': uid,
                'cloudinary_public_id': p.get('cloudinary_public_id'),
                'cloudinary_url': p.get('cloudinary_url'),
            })

    return {
        'version': 1,
        'usuarios': [
            {
                'id': int(u['id']),
                'nome': u['nome'],
                'matricula': u.get('matricula'),
            }
            for u in usuarios
        ],
        'encodings': encodings,
        'photos': photos,
    }


def import_state(db, state: Dict[str, Any]) -> int:
    """Substitui o conteúdo local pelo snapshot. Retorna nº de usuários."""
    if not state or not isinstance(state, dict):
        return 0

    conn = db.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM face_photos')
        cursor.execute('DELETE FROM face_encodings')
        cursor.execute('DELETE FROM usuarios')

        for u in state.get('usuarios') or []:
            cursor.execute(
                'INSERT INTO usuarios (id, nome, matricula) VALUES (?, ?, ?)',
                (int(u['id']), u['nome'], u.get('matricula')),
            )

        for item in state.get('encodings') or []:
            raw = base64.b64decode(item['encoding_b64'])
            cursor.execute(
                'INSERT INTO face_encodings (usuario_id, encoding) VALUES (?, ?)',
                (int(item['usuario_id']), raw),
            )

        for p in state.get('photos') or []:
            cursor.execute(
                'INSERT INTO face_photos (usuario_id, cloudinary_public_id, cloudinary_url) '
                'VALUES (?, ?, ?)',
                (
                    int(p['usuario_id']),
                    p.get('cloudinary_public_id'),
                    p.get('cloudinary_url'),
                ),
            )

        try:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='usuarios'")
            cursor.execute('SELECT MAX(id) AS m FROM usuarios')
            row = cursor.fetchone()
            max_id = row['m'] if row and row['m'] is not None else 0
            if max_id:
                cursor.execute(
                    "INSERT INTO sqlite_sequence(name, seq) VALUES ('usuarios', ?)",
                    (max_id,),
                )
        except Exception:
            pass

        conn.commit()
        return len(state.get('usuarios') or [])
    except Exception as e:
        conn.rollback()
        print(f'Erro ao importar state Cloudinary: {e}')
        return 0
    finally:
        conn.close()


def save_state_to_cloudinary(db) -> bool:
    """Envia snapshot atual para o Cloudinary."""
    if not cloudinary_storage._ensure_cloudinary():
        print('Cloudinary não configurado — state não persistido.')
        return False
    try:
        import cloudinary.uploader
        state = export_state(db)
        payload = json.dumps(state, ensure_ascii=False).encode('utf-8')
        cloudinary.uploader.upload(
            io.BytesIO(payload),
            public_id=STATE_PUBLIC_ID,
            resource_type='raw',
            overwrite=True,
            invalidate=True,
        )
        print(
            f'Cloudinary state salvo: {len(state.get("usuarios") or [])} usuário(s), '
            f'{len(state.get("encodings") or [])} embedding(s).'
        )
        return True
    except Exception as e:
        print(f'Erro ao salvar state no Cloudinary: {e}')
        return False


def load_state_from_cloudinary(db) -> int:
    """Baixa snapshot do Cloudinary e restaura no SQLite."""
    if not cloudinary_storage._ensure_cloudinary():
        return 0
    try:
        import cloudinary.api
        import urllib.request

        info = cloudinary.api.resource(STATE_PUBLIC_ID, resource_type='raw')
        url = info.get('secure_url') or info.get('url')
        if not url:
            return 0
        url = f"{url}{'&' if '?' in url else '?'}_={int(time.time())}"

        with urllib.request.urlopen(url, timeout=45) as resp:
            raw = resp.read().decode('utf-8')
        state = json.loads(raw)
        count = import_state(db, state)
        print(f'Cloudinary state restaurado: {count} usuário(s).')
        return count
    except Exception as e:
        print(f'Cloudinary state não carregado (ok se for o 1º deploy): {e}')
        return 0


def sync_after_change(db, face_recognizer=None) -> None:
    """Persiste no Cloudinary e recarrega embeddings em memória."""
    save_state_to_cloudinary(db)
    if face_recognizer is not None:
        face_recognizer.load_known_faces()
