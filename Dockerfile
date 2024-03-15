# Use the official Python image as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies
RUN pip install --no-cache-dir flask opencv-python-headless numpy

# Expose the port number on which your Flask app will run
EXPOSE 4000

# Run the Flask application
CMD ["python", "black_white_to_color.py",'0.0.0.0:4000']
