# Imagen base de Python
FROM python:3.12-slim

# Update pip
RUN pip3 install --upgrade pip

# Crear un directorio de trabajo
WORKDIR /app

# Install flower
RUN pip3 install flwr>=1.0
RUN pip3 install flwr-datasets>=0.0.2
RUN pip3 install tqdm==4.65.0
RUN pip3 install tensorflow >=2.9.1

# Copiar el archivo del cliente al contenedor
COPY client_1.py /app/client_1.py

# Comando para ejecutar el cliente
CMD ["python", "client_1.py", "--server_address=192.168.1.1:8085", "--cid=0"]

