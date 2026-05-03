FROM python:3.11-slim

WORKDIR /app


# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 7860

# Start from website directory
CMD cd website && gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --timeout 120 --workers 1
