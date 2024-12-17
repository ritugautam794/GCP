# Use the official Python image as a base image
FROM python:3.9-slim

# Set environment variables to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install the necessary Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files to the working directory
COPY . /app/

# Expose the port the app runs on
EXPOSE 8083

# Command to run the Flask app
CMD ["python", "inference-flask.py"]