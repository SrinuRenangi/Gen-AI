"""
=============================================================================
Project: Enterprise AI Deployment on AWS (Day 05)
File: lambda_handler.py
Description: Serverless AWS Lambda Entrypoint using Mangum ASGI Adapter.
             Allows the exact same FastAPI application to run inside AWS Lambda
             behind an Amazon API Gateway or Function URL.
=============================================================================
"""

import logging
from mangum import Mangum
from app import app

logger = logging.getLogger("lambda_handler")
logger.setLevel(logging.INFO)

# Mangum wraps FastAPI into a standard AWS Lambda handler function
# Handles API Gateway HTTP API (payload v2), REST API (payload v1), and ALB events
handler = Mangum(app, lifespan="off")

logger.info("FastAPI Mangum Lambda Handler initialized successfully.")
