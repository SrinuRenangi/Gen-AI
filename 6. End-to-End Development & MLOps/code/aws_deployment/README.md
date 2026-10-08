# 🚀 Enterprise AI Deployment on AWS (Hands-on Project)

> **Zero to Hero GenAI Course — Phase 01: Day 05 Hands-on Project**

This directory contains a complete, production-ready GenAI API microservice engineered for deployment across multiple AWS compute platforms:
- **AWS App Runner** (Fully managed container-to-HTTPS in 3 minutes)
- **AWS ECS Fargate** (Enterprise scalable microservice behind an Application Load Balancer)
- **AWS Lambda** (Serverless Function with Mangum ASGI adapter)
- **Amazon SageMaker** (Dedicated GPU Real-Time Model Endpoint)
- **Amazon Bedrock** (Serverless Foundation Model Gateway with Claude 3.5 Sonnet & Llama 3)

---

## 📁 Project Structure

```
hands_on_project/
├── app.py               # Production FastAPI serving gateway (with streaming & health checks)
├── bedrock_client.py    # Robust Boto3 client wrapper with auto-mock fallback
├── lambda_handler.py    # Mangum ASGI adapter for serverless AWS Lambda execution
├── sagemaker_deploy.py  # Automation script to deploy custom Hugging Face LLMs to SageMaker
├── Dockerfile           # Optimized multi-stage Docker build with non-root security & health checks
├── requirements.txt     # Python dependencies
└── README.md            # Deployment instructions & quickstart guide
```

---

## ⚡ Quickstart: Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python app.py
```
Or with Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to interactive API documentation at:
👉 **`http://localhost:8000/docs`**

*Note: If no AWS credentials are configured, the service automatically runs in **Local Mock Mode**, allowing you to test endpoints, schemas, and streaming without incurring any cloud costs!*

---

## ☁️ Deployment Instructions

### 🚢 Option A: Deploy to AWS App Runner (Fastest)

1. **Build & Tag Docker Image:**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t aws-ai-service .
   docker tag aws-ai-service:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/aws-ai-service:latest
   docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/aws-ai-service:latest
   ```

2. **Create App Runner Service:**
   - In AWS Console ➔ **App Runner** ➔ **Create Service**
   - Source: **Container registry** ➔ Select your ECR image.
   - Instance role: Attach an IAM Role with `AmazonBedrockFullAccess`.
   - Port: `8000`.
   - Click **Deploy**! App Runner provisions HTTPS and auto-scaling automatically.

---

### 📦 Option B: Deploy to Serverless AWS Lambda

1. Create a Lambda function with **Container Image** option.
2. Set CMD to `lambda_handler.handler`.
3. Connect an **Amazon API Gateway HTTP API** with a proxy route (`ANY /{proxy+}`).
4. Add IAM policy `bedrock:InvokeModel` to your Lambda Execution Role.

---

### 🤖 Option C: Deploy Custom LLM to SageMaker

```bash
python sagemaker_deploy.py arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole
```
Deploys a real-time GPU endpoint on `ml.g5.2xlarge` using AWS Deep Learning Containers.
