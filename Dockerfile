FROM python:3.11-slim

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Create a non-root user (optional)
ENV USER=appuser
RUN useradd -ms /bin/bash $USER
WORKDIR /app

# Install Python dependencies first (leveraging Docker layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the port the app runs on (default 8000)
EXPOSE 8000

# Environment variables (overrideable at runtime)
ENV PORT=8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]