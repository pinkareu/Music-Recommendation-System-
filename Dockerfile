# 1. Base image
FROM python:3.11  

# 2. Set working directory inside the container
WORKDIR /backend

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your app code
COPY . .

EXPOSE 8000

# 5. Command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
