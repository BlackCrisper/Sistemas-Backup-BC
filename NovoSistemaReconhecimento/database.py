"""
Módulo para gerenciamento do banco de dados SQLite
"""
import sqlite3
import os
import pickle


class Database:
    def __init__(self, db_path='usuarios.db'):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Retorna uma conexão com o banco de dados"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Inicializa o banco de dados criando as tabelas necessárias"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                matricula TEXT UNIQUE,
                data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                encoding BLOB NOT NULL,
                data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
            )
        ''')

        self._migrate_cpf_to_matricula(cursor)
        self._ensure_matriculas(cursor)

        conn.commit()
        conn.close()

    def _table_columns(self, cursor, table_name):
        cursor.execute(f'PRAGMA table_info({table_name})')
        return {row[1] for row in cursor.fetchall()}

    def _migrate_cpf_to_matricula(self, cursor):
        """Migra coluna cpf → matricula em bancos antigos."""
        cols = self._table_columns(cursor, 'usuarios')
        if 'matricula' in cols:
            return

        if 'cpf' in cols:
            cursor.execute('ALTER TABLE usuarios ADD COLUMN matricula TEXT')
            cursor.execute(
                "UPDATE usuarios SET matricula = printf('%03d', id) "
                "WHERE matricula IS NULL OR matricula = ''"
            )
            # Copia CPF numérico de 3 dígitos se existir e ainda não conflitar
            cursor.execute('SELECT id, cpf FROM usuarios WHERE cpf IS NOT NULL AND cpf != ""')
            for row in cursor.fetchall():
                digits = ''.join(c for c in str(row['cpf']) if c.isdigit())
                if len(digits) == 3:
                    cursor.execute(
                        'UPDATE usuarios SET matricula = ? WHERE id = ? '
                        'AND NOT EXISTS (SELECT 1 FROM usuarios u2 WHERE u2.matricula = ? AND u2.id != ?)',
                        (digits, row['id'], digits, row['id'])
                    )
            try:
                cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_usuarios_matricula ON usuarios(matricula)')
            except sqlite3.OperationalError:
                pass
        else:
            cursor.execute('ALTER TABLE usuarios ADD COLUMN matricula TEXT')
            try:
                cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_usuarios_matricula ON usuarios(matricula)')
            except sqlite3.OperationalError:
                pass

    def _ensure_matriculas(self, cursor):
        """Garante matrícula de 3 dígitos em todos os usuários."""
        cols = self._table_columns(cursor, 'usuarios')
        if 'matricula' not in cols:
            return

        cursor.execute('SELECT id, matricula FROM usuarios ORDER BY id')
        rows = cursor.fetchall()
        used = set()
        for row in rows:
            mat = (row['matricula'] or '').strip()
            if len(mat) == 3 and mat.isdigit():
                used.add(mat)

        next_num = 1
        for row in rows:
            mat = (row['matricula'] or '').strip()
            if len(mat) == 3 and mat.isdigit():
                continue
            while True:
                candidate = f'{next_num:03d}'
                next_num += 1
                if candidate not in used:
                    break
            used.add(candidate)
            cursor.execute('UPDATE usuarios SET matricula = ? WHERE id = ?', (candidate, row['id']))

    @staticmethod
    def normalizar_matricula(matricula):
        if matricula is None:
            return None
        digits = ''.join(c for c in str(matricula) if c.isdigit())
        if len(digits) != 3:
            return None
        return digits

    def criar_usuario(self, nome, matricula=None):
        """
        Cria um novo usuário no banco de dados.
        Retorna o ID do usuário criado ou None se matrícula inválida/duplicada.
        """
        matricula = self.normalizar_matricula(matricula)
        if not matricula:
            return None

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                'INSERT INTO usuarios (nome, matricula) VALUES (?, ?)',
                (nome, matricula)
            )
            usuario_id = cursor.lastrowid
            conn.commit()
            return usuario_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def atualizar_usuario(self, usuario_id, nome, matricula):
        """Atualiza nome e matrícula. Retorna (ok, erro)."""
        matricula = self.normalizar_matricula(matricula)
        if not nome or not nome.strip():
            return False, 'Nome é obrigatório'
        if not matricula:
            return False, 'Matrícula deve ter exatamente 3 dígitos'

        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'UPDATE usuarios SET nome = ?, matricula = ? WHERE id = ?',
                (nome.strip(), matricula, usuario_id)
            )
            if cursor.rowcount == 0:
                return False, 'Usuário não encontrado'
            conn.commit()
            return True, None
        except sqlite3.IntegrityError:
            return False, 'Matrícula já está em uso'
        finally:
            conn.close()

    def buscar_usuario(self, usuario_id):
        """Busca um usuário pelo ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM usuarios WHERE id = ?', (usuario_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def buscar_usuario_por_matricula(self, matricula):
        """Busca um usuário pela matrícula"""
        matricula = self.normalizar_matricula(matricula)
        if not matricula:
            return None

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM usuarios WHERE matricula = ?', (matricula,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def listar_usuarios(self):
        """Lista todos os usuários cadastrados"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM usuarios ORDER BY nome')
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def foto_usuario(self, usuario_id):
        """
        Retorna o nome do primeiro arquivo de foto do usuário, ou None.
        """
        fotos = self.listar_fotos(usuario_id)
        return fotos[0] if fotos else None

    def listar_fotos(self, usuario_id):
        """Lista nomes de arquivos de foto do usuário."""
        user_dir = os.path.join('faces', str(usuario_id))
        if not os.path.isdir(user_dir):
            return []
        return sorted(
            f for f in os.listdir(user_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))
        )

    def limpar_encodings(self, usuario_id):
        """Remove todos os encodings faciais do usuário."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM face_encodings WHERE usuario_id = ?', (usuario_id,))
        conn.commit()
        conn.close()

    def limpar_fotos(self, usuario_id):
        """Remove arquivos de foto do usuário (mantém a pasta)."""
        user_dir = os.path.join('faces', str(usuario_id))
        if not os.path.isdir(user_dir):
            return
        for name in self.listar_fotos(usuario_id):
            try:
                os.remove(os.path.join(user_dir, name))
            except OSError:
                pass

    def adicionar_encoding(self, usuario_id, encoding):
        """
        Adiciona um encoding facial para um usuário
        encoding deve ser um numpy array
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        encoding_blob = pickle.dumps(encoding)

        cursor.execute(
            'INSERT INTO face_encodings (usuario_id, encoding) VALUES (?, ?)',
            (usuario_id, encoding_blob)
        )
        conn.commit()
        conn.close()

    def buscar_encodings_usuario(self, usuario_id):
        """
        Busca todos os encodings de um usuário
        Retorna lista de numpy arrays
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT encoding FROM face_encodings WHERE usuario_id = ?',
            (usuario_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        encodings = []
        for row in rows:
            encoding = pickle.loads(row['encoding'])
            encodings.append(encoding)

        return encodings

    def buscar_todos_encodings(self):
        """
        Busca todos os encodings com informações do usuário
        Retorna lista de dicionários: {usuario_id, nome, encoding}
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT u.id as usuario_id, u.nome, fe.encoding
            FROM face_encodings fe
            JOIN usuarios u ON fe.usuario_id = u.id
        ''')
        rows = cursor.fetchall()
        conn.close()

        encodings_data = []
        for row in rows:
            encoding = pickle.loads(row['encoding'])
            encodings_data.append({
                'usuario_id': row['usuario_id'],
                'nome': row['nome'],
                'encoding': encoding
            })

        return encodings_data

    def deletar_usuario(self, usuario_id):
        """Deleta um usuário e todos os seus encodings"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM face_encodings WHERE usuario_id = ?', (usuario_id,))
        cursor.execute('DELETE FROM usuarios WHERE id = ?', (usuario_id,))
        conn.commit()
        conn.close()

        user_dir = os.path.join('faces', str(usuario_id))
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
