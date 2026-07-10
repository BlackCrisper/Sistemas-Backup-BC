---
name: Camera e UI enxuta
overview: Corrigir o fluxo da câmera (erros silenciosos + liveness bloqueando frames) e redesenhar as interfaces para um UX enxuto e profissional. O algoritmo de reconhecimento fica para a próxima etapa.
todos:
  - id: fix-liveness-stream
    content: Nao bloquear POST /recognize por liveness; feedback de erro no JS do stream
    status: completed
  - id: fix-camera-ux
    content: Robustecer getUserMedia/play e mensagens inline em register e recognize
    status: completed
  - id: ui-redesign
    content: Redesenhar base/index/register/recognize + style.css enxuto e profissional
    status: completed
  - id: validate-camera
    content: Validar que camera abre e stream deixa de spammar 400
    status: completed
isProject: false
---

# Plano: Câmera + UI enxuta

## Diagnóstico (por que “não abre” / não funciona)

Pelos logs do terminal, a câmera **já chegou a abrir**: há dezenas de `POST /recognize` por segundo. O problema real é outro:

```mermaid
flowchart LR
  Webcam --> Frame
  Frame --> POST["POST /recognize"]
  POST --> Liveness["liveness_detector.validate_liveness"]
  Liveness -->|score menor que 0.6| Err400["HTTP 400"]
  Err400 --> Silent["JS ignora erro"]
  Silent --> SemOverlay["Nada aparece na tela"]
```

1. **Liveness muito agressivo** em [`liveness_detector.py`](liveness_detector.py): threshold `combined_score > 0.6` e heurísticas (bordas, FFT, textura) marcam webcam real como “foto”. Em [`app.py`](app.py) (linhas 231–236) isso vira `400`.
2. **Erro silencioso no front** em [`templates/recognize.html`](templates/recognize.html): `recognizeImageRealtime` só trata `data.success`; em 400 não mostra mensagem.
3. **UX frágil da câmera**: sem checagem de `mediaDevices`, sem `video.play()`, feedback só via `alert`, e no cadastro o loop a cada 100ms chama `/api/detect-detailed` (pesado).

O reconhecimento em si (matching de faces) fica para depois, como você pediu — mas **sem destravar o liveness no stream, a câmera continua “morta” na prática**.

## Escopo desta etapa

- Destravar câmera + feedback claro de erro
- Não bloquear o stream de reconhecimento por liveness falso-positivo
- UI enxuta (só o necessário)
- **Não** mexer ainda no algoritmo de matching (`face_recognizer_advanced.py`)

## 1. Câmera e fluxo em tempo real

**Backend** ([`app.py`](app.py)):
- Em `POST /recognize` (stream), **não rejeitar** por liveness; no máximo anexar `liveness` no JSON como aviso.
- Manter liveness só no **cadastro** (`/register`), com threshold mais permissivo se necessário.

**Frontend** ([`templates/recognize.html`](templates/recognize.html), [`templates/register.html`](templates/register.html)):
- Helper único de câmera: checar `navigator.mediaDevices`, pedir permissão, `await video.play()`, mensagens inline (sem `alert`).
- Tratar erros comuns: `NotAllowedError`, `NotFoundError`, `NotReadableError`.
- No reconhecimento: se resposta não for OK, mostrar status na UI; evitar flood (intervalo ~1–1.5s + flag `isProcessing`).
- No cadastro: reduzir polling (ex.: 400–500ms) e usar `/api/detect` em vez de `/api/detect-detailed` no guia contínuo.

## 2. UI/UX enxuta e profissional

Arquivos: [`templates/base.html`](templates/base.html), [`templates/index.html`](templates/index.html), [`templates/register.html`](templates/register.html), [`templates/recognize.html`](templates/recognize.html), [`static/style.css`](static/style.css).

Direção visual (compatível com app interno, não landing):
- Tipografia limpa (ex.: DM Sans via Google Fonts), fundo neutro com leve gradiente
- Nav mínima: logo + 3 links
- Remover: emojis, textos longos de marketing, painel “Informações de Detecção”, footer pesado
- Home: 2 ações (Cadastrar / Reconhecer) + lista compacta de usuários
- Cadastro: nome, CPF opcional, área de câmera, capturar/refazer, cadastrar
- Reconhecer: iniciar/parar câmera, upload, vídeo + status do resultado

## 3. Fora desta etapa

- Debug do matching / embeddings / threshold de similaridade
- Refator grande do YOLO vs OpenCV no detector

## Ordem de implementação

1. Ajustar liveness no `POST /recognize` + mensagens de erro no JS
2. Robustecer `getUserMedia` / `play()` nas duas páginas
3. Redesenhar templates + CSS enxuto
4. Validar manualmente: câmera abre, frames não voltam 400 em massa, UI limpa
