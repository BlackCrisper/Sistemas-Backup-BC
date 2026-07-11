# Deploy — Face ID (microserviço)

Serviço Python separado do backend Node do RefeiControl.

## Pré-requisitos

1. Conta [Cloudinary](https://cloudinary.com) (Cloud name, API Key, API Secret)
2. Conta Render (ou similar) com disco persistente recomendado (≥1 GB RAM)
3. Backend RefeiControl já no ar

## Variáveis do Face ID (Render)

| Variável | Exemplo | Notas |
|----------|---------|--------|
| `FACE_API_KEY` | string longa aleatória | Mesma key no Express |
| `REQUIRE_FACE_API_KEY` | `1` | Obriga a key em produção |
| `FACE_DATA_DIR` | `/var/data` | Volume persistente (SQLite + embeddings) |
| `CLOUDINARY_CLOUD_NAME` | `xxx` | |
| `CLOUDINARY_API_KEY` | `xxx` | |
| `CLOUDINARY_API_SECRET` | `xxx` | |
| `FACE_MATCH_THRESHOLD` | `0.80` | Limiar 1:N |
| `PORT` | automático no Render | |

Use o [`render.yaml`](render.yaml) deste projeto ou crie um Web Service apontando para a pasta `NovoSistemaReconhecimento`:

- **Build:** `pip install -r requirements.txt`
- **Start:** `gunicorn api_server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`

## Variáveis do Express (RefeiControl backend)

No serviço `refeicontrol-backend2` (ou local `.env`):

```
FACE_SERVICE_URL=https://refeicontrol-face-id.onrender.com
FACE_API_KEY=<mesma key do Face ID>
FACE_SERVICE_TIMEOUT_MS=30000
FACE_RECOGNIZE_RATE_MAX=30
```

## Smoke test

```bash
# Health do Face ID
curl https://SEU-FACE-ID.onrender.com/health

# Via proxy do RefeiControl
curl https://SEU-BACKEND.onrender.com/api/face/health
```

Fluxo manual:

1. Admin → Adicionar integrante → capturar 2–3 fotos → salvar
2. Tela pública → Sou integrante → câmera deve reconhecer
3. Botão **Digitar matrícula** deve abrir o fluxo antigo

## Local

```bash
# Terminal 1 — Face ID
cd NovoSistemaReconhecimento
pip install -r requirements.txt
set FACE_API_KEY=change-me-face-api-key
set REQUIRE_FACE_API_KEY=0
python api_server.py

# Terminal 2 — Express
cd refeicontrol-teste/backend
# FACE_SERVICE_URL=http://localhost:5001
# FACE_API_KEY=change-me-face-api-key
npm run dev
```

Cloudinary é opcional em dev (embeddings funcionam sem upload); em produção configure para não perder as fotos de preview.
