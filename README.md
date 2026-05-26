# distributed-cv-engine

A high-performance, distributed computing system for computer vision and 3D geometric processing.

## Description
This system separates the web server API from heavy image calculations. When a user uploads an image or a 3D point cloud, the web application accepts it instantly and delegates the heavy mathematical computations to background workers. This design optimization keeps the application responsive and prevents the main event loop from freezing.

## Tech Stack
* Language: Python
* Computer Vision and Mathematics: NumPy, OpenCV, CuPy (CUDA GPU Acceleration)
* Web API: FastAPI
* Background Task Queue: Celery with Redis Broker
* Containerization: Docker

## Project Structure
* api/ : Contains the web server configuration and API routes (FastAPI).
* workers/ : Contains the background calculation tasks (Celery).
* sandbox/ : Local repository for testing, benchmarking, and optimizing computer vision algorithms.

## 7-Day Sprint Progress
* Day 1 (Matrix Vectorization): Optimizing matrix execution paths using NumPy to eliminate iterative processing loops. 
* Day 2 (Projective Geometry): Implementing camera calibration models and pose estimation frameworks.
* Day 3 (3D Registration): Designing surface tracking, descriptor extraction, and point cloud alignment.
* Day 4 (Async API): Managing high-throughput binary image streams with FastAPI.
* Day 5 (Distributed Workers): Building the pipeline interface between the API and Celery tasks.
* Day 6 (Model Optimization): Quantizing and accelerating neural network inference paths.
* Day 7 (Docker): Packaging and orchestrating the multi-container environment.