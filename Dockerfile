# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txtqq
# We also install ffmpeg and libgl1 which are system dependencies for OpenCV
RUN apt-get update && apt-get install -y ffmpeg libgl1
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Command to run the application using uvicorn web server
# It will be accessible on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]