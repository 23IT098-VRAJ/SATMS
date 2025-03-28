FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the application port (if your app runs on a specific port, e.g., 5000)
EXPOSE 5000

# Command to run the application (adjust according to your entry script)
CMD ["python", "main.py"]
