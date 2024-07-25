import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.messages.ai import AIMessage
import chainlit as cl
import json
from literalai import LiteralClient
from openai import OpenAI
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Inicializar el cliente de Literal
lai = LiteralClient(api_key="lsk_19aKz1WKnUwrsE2Ue7vCNWO0uMXhHgAUztGqbdeQSB4")
lai.instrument_openai()

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Cargar variables de entorno
dotenv_path = Path(r'C:\Users\samuc\OneDrive\Escritorio\Chatbot\.env')
load_dotenv(dotenv_path=dotenv_path)

os.environ["CHAINLIT_AUTH_SECRET"] = "your_chainlit_auth_secret"

# Configuración de los embeddings usando OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Error al cargar OPENAI_API_KEY desde el archivo .env")
model_name = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

# Configuración del modelo de lenguaje de OpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Modelo de OpenAI deseado

class OpenAILLM:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt):
        response = self.llm.invoke([HumanMessage(content=prompt)])
        print(f"Respuesta del modelo: {response}")  # Imprimir la respuesta del modelo
        
        # Validar la respuesta del modelo
        if isinstance(response, list) and response[0].content:
            return response[0].content  # Retornar solo el contenido del primer mensaje
        elif isinstance(response, AIMessage) and response.content:
            return response.content  # Retornar el contenido directamente si es un AIMessage
        else:
            # Depuración adicional para entender mejor la respuesta del modelo
            print(f"Tipo de respuesta: {type(response)}, Contenido: {response}")
            raise ValueError(f"La respuesta del modelo no es válida: {response}")


llm = OpenAILLM(llm=llm)

# Función para cargar vectores desde archivos locales
def load_vectors_from_files():
    vector_files_dir = Path(r'C:\Users\samuc\OneDrive\Escritorio\Chatbot')
    vectors = {}
    vector_file_path = vector_files_dir / 'Requerimientos practicum (1).txt'
    if not vector_file_path.exists():
        print(f"Archivo no encontrado: {vector_file_path}")
        return vectors
    
    with open(vector_file_path, "r", encoding="utf-8") as file:
        content = file.read()
        # Dividir el contenido en párrafos o secciones
        sections = content.split('\n\n')  # Ajusta esto según la estructura de tu documento
        for i, section in enumerate(sections):
            vector_id = f"section_{i}"
            vectors[vector_id] = section.strip()
    return vectors

local_vectors = load_vectors_from_files()

class CustomRetrievalQA:
    def __init__(self, llm, vectors, embeddings):
        self.llm = llm
        self.vectors = vectors
        self.embeddings = embeddings
        self.vector_embeddings = self.precompute_embeddings()

    def precompute_embeddings(self):
        return {vid: self.embeddings.embed_query(text) for vid, text in self.vectors.items()}

    def run(self, query):
        relevant_vectors = self.find_relevant_vectors(query)
        
        if not relevant_vectors:
            return "Lo siento, no encontré información relevante en los documentos."

        context = "\n".join(relevant_vectors)
        prompt = f"Contexto:\n{context}\n\nConsulta: {query}\n\nRespuesta:"
        
        answer = self.llm.generate(prompt)
        return answer

    def find_relevant_vectors(self, query, top_k=3):
        query_embedding = self.embeddings.embed_query(query)
        
        similarities = {}
        for vid, vec_embedding in self.vector_embeddings.items():
            similarity = cosine_similarity([query_embedding], [vec_embedding])[0][0]
            similarities[vid] = similarity
        
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        relevant_vectors = []
        for vid, _ in sorted_similarities[:top_k]:
            relevant_vectors.append(self.vectors[vid])
        
        return relevant_vectors

# Callback de autenticación
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Aquí deberías verificar el nombre de usuario y la contraseña contra tu base de datos
    if (username, password) == ("admin", "admin"):  # Ejemplo estático
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )

# Pre-inicializar chain fuera de on_chat_start
chain = CustomRetrievalQA(llm=llm, vectors=local_vectors, embeddings=embeddings)

@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    welcome_message = f"Hola {app_user.identifier}. Bienvenido al ChatBot de la Camara de Comercio de Loja. Puedes hacerme cualquier pregunta."
    await cl.Message(content=welcome_message).send()
    
    # Forzar una actualización del UI
    await cl.sleep(1)  # Pequeña pausa para asegurar que el mensaje se haya enviado
    

@cl.on_message
async def handle_message(message: cl.Message):
    query = message.content
    logging.debug(f"Query recibida: {query}")
    response = chain.run(query)
    logging.debug(f"Respuesta generada: {response}")
    await cl.Message(content=response).send()

if __name__ == '__main__':
    cl.run()