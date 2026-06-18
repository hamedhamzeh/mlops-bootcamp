# HW03 Docker Image Size Report

| Repository           | Tag       | Size   |
| :------------------- | :-------- | :----- |
| qbc12-airbnb-serving | optimized | 1.29GB |
| qbc12-airbnb-serving | naive     | 2.72GB |

## Analysis

The naive image is larger because it starts from the full Python base image and copies the whole project into the container. This can include files that are not needed at runtime.

The optimized image uses a multi-stage build with python:3.11-slim. The builder stage installs dependencies, while the final runtime stage keeps only the installed packages and application source code.

For production, I would use the optimized image because it is smaller, faster to pull, easier to scan, and has less unnecessary content.
