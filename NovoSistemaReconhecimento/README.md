# Sistema de Reconhecimento Facial com YOLO

Sistema de reconhecimento facial desenvolvido em Python utilizando YOLOv8 para detecção de rostos e a biblioteca `face_recognition` para identificação de pessoas.

## Características

- **Detecção de Rostos**: Utiliza YOLOv8 para detectar pessoas e extrair regiões faciais
- **Reconhecimento Facial**: Compara faces detectadas com banco de dados usando embeddings
- **Interface Web**: Interface moderna e responsiva desenvolvida com Flask
- **Cadastro de Usuários**: Sistema para adicionar novas faces ao banco de dados
- **Reconhecimento em Tempo Real**: Suporte para webcam e upload de imagens
- **API REST**: Endpoints para integração com outros sistemas

## Requisitos

- Python 3.8 ou superior
- Webcam (opcional, para reconhecimento em tempo real)

## Instalação

1. Clone ou baixe este repositório

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

**✅ INSTALAÇÃO SIMPLES - Sem Visual Studio!**

O sistema agora usa **apenas OpenCV**, que não precisa de compilação! 

**Instalação em 3 passos:**
1. `pip install -r requirements.txt` (sem dlib, sem compilação!)
2. `python app.py`
3. Acesse `http://localhost:5000`

**Consulte `INSTALACAO_SIMPLES.md` para mais detalhes.**

O sistema usa OpenCV LBPH para reconhecimento facial, que funciona perfeitamente no Windows sem ferramentas adicionais.

## Estrutura do Projeto

```
NovoSistemaReconhecimento/
├── app.py                 # Aplicação Flask principal
├── requirements.txt       # Dependências Python
├── database.py            # Gerenciamento do banco de dados SQLite
├── face_detector.py       # Detecção de rostos com YOLO
├── face_recognizer.py     # Reconhecimento facial com embeddings
├── models/                # Modelos YOLO (baixados automaticamente)
├── faces/                 # Diretório para armazenar faces cadastradas
├── static/                # Arquivos estáticos (CSS, JS)
├── templates/             # Templates HTML
└── usuarios.db            # Banco de dados SQLite (criado automaticamente)
```

## Como Usar

1. **Inicie o servidor Flask**:
```bash
python app.py
```

2. **Acesse a interface web**:
Abra seu navegador em `http://localhost:5000`

3. **Cadastre uma nova face**:
   - Clique em "Cadastrar Face"
   - Preencha o nome (e opcionalmente o CPF)
   - Faça upload de uma imagem ou capture uma foto usando a webcam
   - Clique em "Cadastrar"

4. **Reconheça faces**:
   - Clique em "Reconhecer"
   - Inicie a webcam ou faça upload de uma imagem
   - O sistema identificará pessoas cadastradas

## API REST

### Listar Usuários
```
GET /api/users
```

### Reconhecimento via API
```
POST /api/recognize
Content-Type: multipart/form-data

Body: imagem (arquivo de imagem)
```

### Deletar Usuário
```
DELETE /api/users/<user_id>
```

## Tecnologias Utilizadas

- **YOLOv8 (Ultralytics)**: Detecção de pessoas e objetos
- **face_recognition**: Extração de embeddings e comparação de faces
- **Flask**: Framework web
- **OpenCV**: Processamento de imagens
- **SQLite**: Banco de dados
- **NumPy**: Operações numéricas

## Notas Importantes

- O modelo YOLO será baixado automaticamente na primeira execução
- A biblioteca `face-recognition` usa modelos pré-treinados baseados em dlib
- O sistema armazena encodings faciais no banco de dados para comparação rápida
- A tolerância padrão para reconhecimento é 0.6 (pode ser ajustada em `face_recognizer.py`)

## Solução de Problemas

### Erro ao instalar face-recognition
Se encontrar problemas ao instalar `face-recognition`, tente:
```bash
pip install cmake
pip install dlib-binary
pip install face-recognition
```

### Webcam não funciona
Certifique-se de que:
- A webcam está conectada e funcionando
- O navegador tem permissão para acessar a webcam
- Você está usando HTTPS ou localhost (navegadores modernos exigem)

### Nenhum rosto detectado
- Certifique-se de que a imagem contém um rosto visível
- Tente com diferentes ângulos e iluminação
- Verifique se a qualidade da imagem é adequada

## Licença

Este projeto é fornecido como está, para fins educacionais e de desenvolvimento.
