FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code
COPY . .

# Expose the port Hugging Face Spaces uses
EXPOSE 7860

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]