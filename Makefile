# Define variables
IMAGE_NAME = multi-task-transformer
CONTAINER_NAME = CONTAINER_NAME = multi-task-container

# Default target
all: build run

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
	@echo "Running Docker container..."
	docker run -it --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Stop and remove the Docker container
stop:
	@echo "Stopping Docker container..."
	docker stop $(CONTAINER_NAME) || true
	@echo "Removing Docker container..."
	docker rm $(CONTAINER_NAME) || true

# Clean up Docker images and containers
clean:
	@echo "Cleaning up Docker images and containers..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME) || true

# Show help
help:
	@echo "Makefile for Sentence Transformer & Multi-Task Learning"
	@echo ""
	@echo "Usage:"
	@echo "  make build    - Build the Docker image"
	@echo "  make run      - Run the Docker container"
	@echo "  make stop     - Stop and remove the Docker container"
	@echo "  make clean    - Clean up Docker images and containers"
	@echo "  make help     - Show this help message"
