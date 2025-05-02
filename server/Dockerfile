FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/metrics /app/logs /app/models /app/data

# Generate certificates if needed
RUN mkdir -p /app/certificates
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /app/certificates/server.key \
    -out /app/certificates/server.crt \
    -subj "/CN=localhost" \
    || echo "Certificate generation skipped - add your own certificates"

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV CONFIG_PATH=/app/config.yaml
ENV LOG_LEVEL=INFO
ENV CERTIFICATES_DIR=/app/certificates
ENV METRICS_DIR=/app/metrics

# Create a non-root user
RUN useradd -m fids
RUN chown -R fids:fids /app
USER fids

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

# Expose the server port
EXPOSE 8080

# Command to run the server
CMD ["python", "-m", "fids.server", "--config", "${CONFIG_PATH}", "--certs", "${CERTIFICATES_DIR}", "--metrics", "${METRICS_DIR}"]