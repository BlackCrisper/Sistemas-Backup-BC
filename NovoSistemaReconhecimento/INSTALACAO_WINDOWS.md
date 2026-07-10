# Guia de Instalação para Windows

## Problema com dlib

O `dlib` e `face-recognition` requerem compilação no Windows, o que pode ser complicado. Siga uma das opções abaixo:

## Opção 1: Instalar CMake (Recomendado)

1. **Baixe e instale o CMake:**
   - Acesse: https://cmake.org/download/
   - Baixe a versão "Windows x64 Installer"
   - Durante a instalação, **marque a opção "Add CMake to system PATH"**

2. **Instale o Visual Studio Build Tools:**
   - Acesse: https://visualstudio.microsoft.com/downloads/
   - Baixe "Build Tools for Visual Studio"
   - Durante a instalação, selecione "Desktop development with C++"

3. **Reinicie o terminal/PowerShell**

4. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Opção 2: Usar Wheel Pré-compilada (Mais Rápido)

1. **Baixe uma wheel pré-compilada do dlib:**
   - Acesse: https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases
   - Ou use: https://github.com/sachadee/Dlib/releases
   - Baixe a wheel compatível com sua versão do Python (ex: `dlib-19.24.0-cp311-cp311-win_amd64.whl` para Python 3.11)

2. **Instale a wheel:**
   ```bash
   pip install caminho/para/dlib-19.24.0-cp311-cp311-win_amd64.whl
   ```

3. **Instale o restante:**
   ```bash
   pip install face-recognition flask ultralytics opencv-python numpy pillow
   ```

## Opção 3: Usar Conda (Alternativa)

Se você tem Anaconda ou Miniconda instalado:

```bash
conda install -c conda-forge dlib
pip install face-recognition flask ultralytics opencv-python numpy pillow
```

## Opção 4: Usar Docker (Mais Complexo)

Se você tem Docker instalado, pode usar uma imagem Linux pré-configurada.

## Verificação

Após a instalação, teste se funcionou:

```python
python -c "import dlib; import face_recognition; print('OK!')"
```

Se não houver erros, a instalação foi bem-sucedida!
