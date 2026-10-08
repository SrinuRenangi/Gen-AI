# =============================================================================
# Production Dockerfile: AWS AI Microservice
# Compatible with AWS App Runner, AWS ECS (Fargate), and Local Docker
# =============================================================================

# Step 1: Base image using official Python slim
FROM python:3.11-slim as base

# Set environment variables for production performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Step 2: Install system dependencies & curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy application source code
COPY bedrock_client.py app.py lambda_handler.py ./

# Step 5: Create a non-root user for security best practices
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose standard container serving port
EXPOSE 8000

# Step 6: Container Health Check Probe for AWS Application Load Balancers
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Step 7: Launch FastAPI via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
