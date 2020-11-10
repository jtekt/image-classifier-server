# Using a prebuilt image to save on building time
#FROM python:3.8
FROM jjanzic/docker-python3-opencv

# Create app directory and move into it
WORKDIR /usr/src/app

# Copy all files into container
COPY . .

# Install python modules
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 80

# Run the app
CMD [ "gunicorn", "server:app", "-b 0.0.0.0:80"]
