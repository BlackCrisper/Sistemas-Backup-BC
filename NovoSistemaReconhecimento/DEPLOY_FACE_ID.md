# Deploy — Face ID (microserviço) sem Disk pago

Persistência: **Cloudinary** (fotos + JSON com embeddings). Não precisa de Disk no Render.

## Variáveis no Render (Face ID)

```
REQUIRE_FACE_API_KEY=1
FACE_API_KEY=<mesma do backend Node>
FACE_MATCH_THRESHOLD=0.70
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
```

**Não defina** `FACE_DATA_DIR` (evita erro de `/var/data`).

## Backend Node

```
FACE_SERVICE_URL=https://SEU-FACE-ID.onrender.com
FACE_API_KEY=<mesma key>
```

## Fluxo

1. Boot → baixa `refeicontrol/face-db/state` do Cloudinary → reconstrói SQLite em `/tmp`
2. Enroll/delete → sobe fotos + atualiza o JSON de state no Cloudinary

