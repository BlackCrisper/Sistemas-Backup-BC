"""
Upload e remoção de fotos faciais no Cloudinary.
Pasta por usuário: refeicontrol/faces/{matricula}-{nome}
"""
import os
import re
from typing import List, Optional, Tuple

_configured = False


def is_configured() -> bool:
    return bool(
        os.environ.get('CLOUDINARY_CLOUD_NAME')
        and os.environ.get('CLOUDINARY_API_KEY')
        and os.environ.get('CLOUDINARY_API_SECRET')
    )


def _ensure_cloudinary():
    global _configured
    if _configured:
        return True
    if not is_configured():
        return False
    import cloudinary
    cloudinary.config(
        cloud_name=os.environ['CLOUDINARY_CLOUD_NAME'],
        api_key=os.environ['CLOUDINARY_API_KEY'],
        api_secret=os.environ['CLOUDINARY_API_SECRET'],
        secure=True,
    )
    _configured = True
    return True


def _slug_nome(nome: str) -> str:
    raw = (nome or '').strip()
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', raw)
    slug = re.sub(r'_+', '_', slug).strip('_')[:40]
    return slug


def folder_for_user(matricula: str, nome: str = '') -> str:
    """Pasta dedicada no Cloudinary para o usuário."""
    mat = str(matricula or '').strip()
    slug = _slug_nome(nome)
    if slug:
        return f'refeicontrol/faces/{mat}-{slug}'
    return f'refeicontrol/faces/{mat}'


def upload_face_image(
    image_bytes: bytes,
    matricula: str,
    index: int,
    nome: str = '',
) -> Optional[Tuple[str, str]]:
    """
    Faz upload de uma imagem JPEG/PNG na pasta do usuário.
    Retorna (public_id, secure_url) ou None se Cloudinary não estiver configurado/falhar.
    """
    if not _ensure_cloudinary():
        return None
    try:
        import cloudinary.uploader
        folder = folder_for_user(matricula, nome)
        result = cloudinary.uploader.upload(
            image_bytes,
            folder=folder,
            public_id=f'sample_{index}',
            overwrite=True,
            resource_type='image',
            format='jpg',
            tags=[f'matricula:{matricula}', 'refeicontrol-face'],
            context=f'matricula={matricula}|nome={nome or matricula}',
        )
        return result.get('public_id'), result.get('secure_url')
    except Exception as e:
        print(f'Cloudinary upload error: {e}')
        return None


def delete_folder_for_matricula(matricula: str, nome: str = '') -> int:
    """Remove recursos da pasta do usuário (padrão novo e antigo)."""
    if not _ensure_cloudinary():
        return 0
    try:
        import cloudinary.api
        mat = str(matricula or '').strip()
        # Prefixo cobre faces/{mat}/... e faces/{mat}-Nome/...
        prefix = f'refeicontrol/faces/{mat}'
        result = cloudinary.api.delete_resources_by_prefix(prefix)
        deleted = result.get('deleted') or {}
        return len(deleted)
    except Exception as e:
        print(f'Cloudinary delete error: {e}')
        return 0


def list_urls_for_user(photos: List[dict]) -> List[str]:
    return [p['cloudinary_url'] for p in photos if p.get('cloudinary_url')]
