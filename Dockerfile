FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-m", "zlite"]
CMD ["order", "--structure", "examples/structure.json", "-n", "20", "-o", "/app/out/demo"]
