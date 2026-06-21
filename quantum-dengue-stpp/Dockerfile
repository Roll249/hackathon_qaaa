# =============================================================================
# Quantum Dengue STPP - Docker Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Python Dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as requirements

WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional runtime dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    uvicorn[standard]>=0.30 \
    pydantic>=2.9

# -----------------------------------------------------------------------------
# Stage 2: Testing Environment
# -----------------------------------------------------------------------------
FROM python:3.11-slim as test

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest pytest-cov pytest-xdist

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src"]

# -----------------------------------------------------------------------------
# Stage 3: Production API Server
# -----------------------------------------------------------------------------
FROM python:3.11-slim as api

LABEL maintainer="Quantum Dengue Team"
LABEL description="Quantum-Enhanced Dengue Forecasting API"

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy virtual environment from builder
COPY --from=requirements /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY dengue_dataset/ ./dengue_dataset/

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run API server
CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 4: GPU Production Server
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu-api

LABEL maintainer="Quantum Dengue Team"
LABEL description="Quantum-Enhanced Dengue Forecasting GPU API"

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
RUN mkdir -p /app/logs && chown -R appuser:appgroup /app

USER appuser
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 5: Jupyter Notebook for Research
# -----------------------------------------------------------------------------
FROM jupyter/scipy-notebook:latest as notebook

LABEL maintainer="Quantum Dengue Team"
LABEL description="Quantum Dengue Research Environment"

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
USER root
RUN pip install --no-cache-dir -r requirements.txt
USER $NB_UID

# Copy notebooks and data
COPY notebooks/ ./notebooks/
COPY dengue_dataset/ ./dengue_dataset/

EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''"]

# -----------------------------------------------------------------------------
# Stage 6: Full Multi-stage Build (Default)
# -----------------------------------------------------------------------------
FROM python:3.11-slim as final

WORKDIR /app

# Copy from other stages as needed
COPY --from=requirements /opt/venv /opt/venv
COPY src/ ./src/
COPY dengue_dataset/ ./dengue_dataset/
COPY output_result/ ./output_result/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-c", "print('Quantum Dengue STPP Docker Container')"]
