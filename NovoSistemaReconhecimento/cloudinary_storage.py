"""
Upload e remoção de fotos faciais no Cloudinary.
Se as credenciais não estiverem configuradas, as operações são no-op (embeddings ainda funcionam).
"""
import os
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


def upload_face_image(image_bytes: bytes, matricula: str, index: int) -> Optional[Tuple[str, str]]:
    """
    Faz upload de uma imagem JPEG/PNG.
    Retorna (public_id, secure_url) ou None se Cloudinary não estiver configurado/falhar.
    """
    if not _ensure_cloudinary():
        return None
    try:
        import cloudinary.uploader
        folder = f"refeicontrol/faces/{matricula}"
        result = cloudinary.uploader.upload(
            image_bytes,
            folder=folder,
            public_id=f"sample_{index}",
            overwrite=True,
            resource_type='image',
            format='jpg',
        )
        return result.get('public_id'), result.get('secure_url')
    except Exception as e:
        print(f'Cloudinary upload error: {e}')
        return None


def delete_folder_for_matricula(matricula: str) -> int:
    """Remove recursos da pasta do usuário. Retorna quantidade aproximada removida."""
    if not _ensure_cloudinary():
        return 0
    try:
        import cloudinary.api
        prefix = f"refeicontrol/faces/{matricula}"
        result = cloudinary.api.delete_resources_by_prefix(prefix)
        deleted = result.get('deleted') or {}
        return len(deleted)
    except Exception as e:
        print(f'Cloudinary delete error: {e}')
        return 0


def list_urls_for_user(photos: List[dict]) -> List[str]:
    return [p['cloudinary_url'] for p in photos if p.get('cloudinary_url')]
