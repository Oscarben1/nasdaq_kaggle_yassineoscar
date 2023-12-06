# Dockerfile

# Utilisez une image de base avec Python
FROM python:3.8

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers nécessaires dans le conteneur
COPY src/app.py /app/
COPY model/testmodel.pkl /app/
# Ajoutez des paquets système nécessaires
RUN apt-get update && \
    apt-get install -y libgomp1

# Copiez le fichier requirements.txt et installez les dépendances
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port sur lequel l'application Flask va écouter
EXPOSE 5001

# Commande pour lancer l'application Flask
CMD ["python", "app.py", "--host=0.0.0.0", "--port=5001"]
