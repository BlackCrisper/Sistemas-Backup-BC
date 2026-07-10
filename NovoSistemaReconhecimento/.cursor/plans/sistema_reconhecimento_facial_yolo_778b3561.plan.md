---
name: Sistema Reconhecimento Facial YOLO
overview: Sistema de reconhecimento facial usando YOLOv8 para detecção de rostos, biblioteca face_recognition para identificação, interface web com Flask, e SQLite para armazenamento de dados dos usuários.
todos:
  - id: setup_project
    content: Criar estrutura de diretórios e arquivo requirements.txt com todas as dependências (ultralytics, face-recognition, flask, opencv-python, numpy, pillow)
    status: completed
  - id: database_module
    content: Implementar database.py com criação de tabelas SQLite (usuarios, face_encodings) e funções CRUD
    status: completed
  - id: face_detector
    content: Implementar face_detector.py usando YOLOv8 para detecção de rostos em imagens e vídeo
    status: completed
  - id: face_recognizer
    content: Implementar face_recognizer.py usando face_recognition para extrair embeddings e comparar faces
    status: completed
  - id: flask_app
    content: Criar app.py com rotas Flask para interface web (index, register, recognize) e API REST
    status: completed
  - id: templates_html
    content: Criar templates HTML (index.html, register.html, recognize.html) com interface moderna e responsiva
    status: completed
  - id: static_files
    content: Criar arquivo CSS para estilização da interface web
    status: completed
  - id: integration_test
    content: "Testar fluxo completo: cadastro de face, detecção e reconhecimento"
    status: completed
---

# Sistema de Reconhecimento Facial com YOLO e Python

## Arquitetura do Sistema

O sistema será composto por:

- **YOLOv8** (Ultralytics): Detecção de rostos em imagens/vídeo
- **face_recognition**: Biblioteca para extração de embeddings e comparação de faces
- **Flask**: Interface web para cadastro e reconhecimento
- **SQLite**: Banco de dados para armazenar informações dos usuários
- **OpenCV**: Processamento de imagens e captura de vídeo

## Estrutura de Arquivos

```
NovoSistemaReconhecimento/
├── app.py                 # Aplicação Flask principal
├── requirements.txt       # Dependências Python
├── database.py            # Gerenciamento do banco de dados SQLite
├── face_detector.py       # Detecção de rostos com YOLO
├── face_recognizer.py     # Reconhecimento facial com embeddings
├── models/                # Modelos YOLO baixados
├── faces/                 # Diretório para armazenar faces cadastradas
│   └── {user_id}/        # Subdiretórios por usuário
├── static/                # Arquivos estáticos (CSS, JS)
│   └── style.css
├── templates/             # Templates HTML
│   ├── index.html         # Página principal
│   ├── register.html      # Cadastro de nova face
│   └── recognize.html     # Página de reconhecimento
└── usuarios.db            # Banco de dados SQLite
```

## Componentes Principais

### 1. Banco de Dados (`database.py`)

- Tabela `usuarios`: id, nome, cpf, data_cadastro
- Tabela `face_encodings`: id, usuario_id, encoding (vetor de características)
- Funções para CRUD de usuários e encodings

### 2. Detector de Rostos (`face_detector.py`)

- Usa YOLOv8 pré-treinado para detecção de rostos
- Retorna coordenadas dos bounding boxes
- Processa imagens e frames de vídeo

### 3. Reconhecedor Facial (`face_recognizer.py`)

- Extrai embeddings usando `face_recognition`
- Compara embeddings com banco de dados
- Retorna identificação com nível de confiança

### 4. Aplicação Flask (`app.py`)

- Rota `/`: Página inicial
- Rota `/register`: Cadastro de nova face (upload de imagem)
- Rota `/recognize`: Reconhecimento em tempo real via webcam
- Rota `/api/recognize`: API para reconhecimento via upload de imagem

## Fluxo de Funcionamento

### Cadastro de Nova Face

1. Usuário faz upload de imagem ou captura via webcam
2. YOLO detecta rosto na imagem
3. Extrai embedding usando `face_recognition`
4. Salva no banco de dados associado ao usuário
5. Armazena imagem original em `faces/{user_id}/`

### Reconhecimento

1. Captura frame da webcam ou recebe imagem
2. YOLO detecta todos os rostos no frame
3. Para cada rosto, extrai embedding
4. Compara com encodings no banco de dados
5. Retorna identificação se confiança > threshold

## Bibliotecas Necessárias

- `ultralytics` (YOLOv8)
- `face-recognition` (baseado em dlib)
- `flask`
- `opencv-python`
- `numpy`
- `pillow`
- `sqlite3` (built-in)

## Funcionalidades Implementadas

1. **Cadastro de Usuário**: Formulário web para cadastrar nome, CPF e upload de foto
2. **Detecção de Rostos**: YOLO detecta rostos em imagens/vídeo
3. **Reconhecimento**: Compara faces detectadas com banco de dados
4. **Interface Web**: Interface amigável para todas as operações
5. **API REST**: Endpoints para integração com outros sistemas