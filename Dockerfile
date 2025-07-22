FROM python:3.12-slim

# Set working directory
WORKDIR /app

# install ssh client
RUN apt update && apt install -y openssh-client && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

ARG USERNAME
ARG UID
ARG GID
RUN groupadd -g $GID $USERNAME \
 && useradd -m -u $UID -g $GID $USERNAME \
 && chown -R $USERNAME:$USERNAME /app

# Switch to the new user
USER $USERNAME

ENV PYTHONUNBUFFERED=1

# Run the worker
CMD ["python", "-m", "app.main"]
