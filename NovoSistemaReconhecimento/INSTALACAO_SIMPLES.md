# Instalação Simplificada - Sem Visual Studio!

## ✅ Solução Mais Simples

Agora o sistema usa **apenas OpenCV**, que não precisa de compilação! 

## Instalação em 3 Passos

### 1. Instale as dependências (sem dlib!)

```bash
pip install -r requirements.txt
```

Isso instalará:
- `ultralytics` (YOLO)
- `opencv-python` e `opencv-contrib-python` (reconhecimento facial)
- `flask` (interface web)
- `numpy`, `pillow` (processamento de imagens)

**Sem necessidade de:**
- ❌ Visual Studio
- ❌ CMake
- ❌ dlib
- ❌ face-recognition

### 2. Execute o sistema

```bash
python app.py
```

### 3. Acesse no navegador

Abra: `http://localhost:5000`

## Como Funciona Agora

O sistema usa:
- **YOLO**: Para detecção de pessoas (opcional, pode usar OpenCV também)
- **OpenCV LBPH**: Para reconhecimento facial (nativo, sem compilação)
- **Haar Cascade**: Para detecção de rostos (já vem no OpenCV)

Tudo funciona direto no Windows sem precisar compilar nada! 🎉

## Teste Rápido

Após instalar, teste se está tudo OK:

```python
python -c "import cv2; import numpy; import flask; print('Tudo OK!')"
```

Se não houver erros, está pronto para usar!
