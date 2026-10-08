"""
=============================================================================
Project: Enterprise AI Deployment on AWS (Day 05)
File: sagemaker_deploy.py
Description: End-to-end Python script to deploy an open-source Hugging Face model
             onto a dedicated Amazon SageMaker Real-Time GPU Endpoint.
             Includes cost safeguards and automated endpoint cleanup.
=============================================================================
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagemaker_deploy")

try:
    import boto3
    import sagemaker
    from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
    HAS_SAGEMAKER = True
except ImportError:
    HAS_SAGEMAKER = False


def deploy_huggingface_llm(
    model_id: str = "HuggingFaceH4/zephyr-7b-beta",
    instance_type: str = "ml.g5.2xlarge",
    number_of_gpu: int = 1,
    role_arn: str = None,
    teardown_after_test: bool = True,
):
    """
    Deploy an LLM from Hugging Face directly onto a SageMaker real-time endpoint
    using AWS Large Model Inference (TGI / vLLM) Deep Learning Containers.
    """
    if not HAS_SAGEMAKER:
        logger.error("sagemaker / boto3 library is not installed. Run: pip install sagemaker boto3")
        return

    logger.info("Initializing SageMaker session...")
    try:
        session = sagemaker.Session()
        region = session.boto_region_name
        role = role_arn or sagemaker.get_execution_role()
    except Exception as exc:
        logger.warning(
            "Could not automatically detect SageMaker execution role: %s. "
            "Please provide a valid 'role_arn' parameter with AmazonSageMakerFullAccess permissions.",
            exc,
        )
        return

    logger.info("Target Region: %s | Execution Role: %s", region, role)

    # 1. Hub Model Configuration for Text Generation Inference (TGI)
    hub_config = {
        "HF_MODEL_ID": model_id,
        "SM_NUM_GPUS": json.dumps(number_of_gpu),
        "MAX_INPUT_LENGTH": json.dumps(2048),
        "MAX_TOTAL_TOKENS": json.dumps(4096),
        "MAX_BATCH_PREFILL_TOKENS": json.dumps(4096),
        "MESSAGES_API_ENABLED": "true",
    }

    # 2. Retrieve the official AWS Deep Learning Container for TGI
    try:
        image_uri = get_huggingface_llm_image_uri(
            "huggingface",
            version="2.0.2",
            session=session,
        )
        logger.info("Retrieved Deep Learning Container: %s", image_uri)
    except Exception as exc:
        logger.error("Failed to retrieve LLM DLC image: %s", exc)
        return

    # 3. Create SageMaker Model Object
    endpoint_name = f"llm-{model_id.split('/')[-1].lower()[:15]}-{int(time.time())}"
    huggingface_model = HuggingFaceModel(
        image_uri=image_uri,
        env=hub_config,
        role=role,
        sagemaker_session=session,
    )

    logger.info("Deploying model to endpoint '%s' on %s...", endpoint_name, instance_type)
    logger.info("Note: Provisioning a GPU instance and downloading weights takes ~5-10 minutes.")

    # 4. Deploy to Real-Time Endpoint
    try:
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            container_startup_health_check_timeout=600,
        )
        logger.info("🎉 SageMaker Endpoint successfully deployed: %s", endpoint_name)

        # 5. Run a live test inference
        test_payload = {
            "inputs": "Explain the architectural difference between Amazon Bedrock and Amazon SageMaker in 3 bullet points:",
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.4,
                "top_p": 0.9,
            },
        }

        logger.info("Sending test inference to SageMaker endpoint...")
        response = predictor.predict(test_payload)
        print("\n" + "=" * 60)
        print("SAGEMAKER INFERENCE RESULT:")
        print("=" * 60)
        print(json.dumps(response, indent=2))
        print("=" * 60 + "\n")

    except Exception as exc:
        logger.error("Error during deployment or inference: %s", exc)
    finally:
        # 6. Teardown safeguards: Prevent unexpected cloud billing!
        if teardown_after_test:
            logger.info("⚠️ TEARDOWN SAFEGUARD: Deleting endpoint '%s' to prevent GPU charges...", endpoint_name)
            try:
                predictor.delete_endpoint()
                predictor.delete_model()
                logger.info("✅ Endpoint and model deleted successfully.")
            except Exception as e:
                logger.error("Failed to delete endpoint: %s. Please delete manually in AWS Console.", e)


if __name__ == "__main__":
    print("--- Amazon SageMaker LLM Automated Deployment Utility ---")
    print("To run live deployment, provide your AWS SageMaker Execution Role ARN:")
    print("Example: python sagemaker_deploy.py arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole")
    
    if len(sys.argv) > 1:
        custom_role = sys.argv[1]
        deploy_huggingface_llm(role_arn=custom_role, teardown_after_test=True)
    else:
        logger.info("No role ARN provided. Run script with role ARN to trigger live deployment.")
