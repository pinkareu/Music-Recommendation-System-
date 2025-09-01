# 1️⃣ Base image: slim version of Python 3.11
FROM python:3.11-slim

# 2️⃣ Set working directory inside the container
WORKDIR /backend

# 3️⃣ Install only necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Copy requirements and install Python dependencies
COPY requirements.txt .

# Install CPU-only PyTorch to save space, then other packages
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy application code into container
COPY . .

# 6️⃣ Expose the port your app will run on
EXPOSE 8000

# 7️⃣ Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
