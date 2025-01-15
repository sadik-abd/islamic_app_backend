# Use a base image with Python 3.10
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app
RUN apt-get update && apt-get install -y wget	git 	git-lfs 	ffmpeg 	libsm6 	libxext6 	cmake 	rsync 	libgl1-mesa-glx 	&& rm -rf /var/lib/apt/lists/* 	&& git lfs install
RUN pip install --no-cache-dir pip==24.3.1 && 	pip install --no-cache-dir 	datasets 	"huggingface-hub>=0.19" "hf-transfer>=0.1.4" "protobuf<4" "click<8.1" "pydantic~=1.0"


# Copy the local requirements file to the container
COPY requirements.txt .


# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN cd faiss_index2 && wget https://github.com/sadik-abd/islamic_app_backend/releases/download/v1/index.faiss
# Set the entrypoint for your application
CMD ["python" ,"api.py"]
