"""
Módulo para gerenciamento do banco de dados SQLite
"""
import sqlite3
import os
from datetime import datetime
import numpy as np
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
        
        # Tabela de usuários
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                cpf TEXT UNIQUE,
                data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de encodings faciais
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                encoding BLOB NOT NULL,
                data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def criar_usuario(self, nome, cpf=None):
        """
        Cria um novo usuário no banco de dados
        Retorna o ID do usuário criado
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO usuarios (nome, cpf) VALUES (?, ?)',
                (nome, cpf)
            )
            usuario_id = cursor.lastrowid
            conn.commit()
            return usuario_id
        except sqlite3.IntegrityError:
            # CPF já existe
            return None
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
    
    def buscar_usuario_por_cpf(self, cpf):
        """Busca um usuário pelo CPF"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM usuarios WHERE cpf = ?', (cpf,))
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
    
    def adicionar_encoding(self, usuario_id, encoding):
        """
        Adiciona um encoding facial para um usuário
        encoding deve ser um numpy array
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Serializa o encoding para BLOB
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
        
        cursor.execute('DELETE FROM usuarios WHERE id = ?', (usuario_id,))
        conn.commit()
        conn.close()
        
        # Remove diretório de imagens do usuário
        user_dir = os.path.join('faces', str(usuario_id))
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
