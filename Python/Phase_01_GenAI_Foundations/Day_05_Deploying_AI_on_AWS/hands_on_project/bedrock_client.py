"""
=============================================================================
Project: Enterprise AI Deployment on AWS (Day 05)
File: bedrock_client.py
Description: Production Boto3 Client Wrapper for Amazon Bedrock Foundation Models.
             Supports Claude 3.5 Sonnet, Llama 3, Titan Embeddings, Streaming,
             and an automatic local mock mode for development without AWS credentials.
=============================================================================
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger("bedrock_client")
logging.basicConfig(level=logging.INFO)

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class BedrockAIClient:
    """Enterprise-ready wrapper for invoking models on Amazon Bedrock."""

    # Default Model IDs on Amazon Bedrock
    MODEL_CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    MODEL_CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    MODEL_LLAMA_3_70B = "meta.llama3-70b-instruct-v1:0"
    MODEL_LLAMA_3_8B = "meta.llama3-8b-instruct-v1:0"
    MODEL_TITAN_EMBEDDING = "amazon.titan-embed-text-v1"

    def __init__(
        self,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 60,
    ):
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.client = None
        self.is_mock = False

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the boto3 bedrock-runtime client with retry policies."""
        if not HAS_BOTO3:
            logger.warning("boto3 not installed. Running BedrockAIClient in LOCAL MOCK MODE.")
            self.is_mock = True
            return

        # Check if AWS credentials exist
        has_creds = bool(
            os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE") or os.path.exists(os.path.expanduser("~/.aws/credentials"))
        )

        if not has_creds:
            logger.warning("No AWS credentials detected in environment. Running BedrockAIClient in LOCAL MOCK MODE.")
            self.is_mock = True
            return

        try:
            retry_config = Config(
                region_name=self.region_name,
                retries={"max_attempts": self.max_retries, "mode": "adaptive"},
                connect_timeout=self.timeout_seconds,
                read_timeout=self.timeout_seconds,
            )
            self.client = boto3.client("bedrock-runtime", config=retry_config)
            logger.info("Successfully connected to Amazon Bedrock client in region: %s", self.region_name)
        except Exception as exc:
            logger.warning("Failed to initialize boto3 client (%s). Falling back to MOCK MODE.", exc)
            self.is_mock = True

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful, enterprise-grade AI assistant running on AWS.",
        model_id: str = MODEL_CLAUDE_3_5_SONNET,
        max_tokens: int = 1000,
        temperature: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Synchronously invoke a Bedrock Foundation Model using the modern Converse API.
        """
        if self.is_mock:
            return {
                "model_id": model_id,
                "text": f"[LOCAL MOCK AWS BEDROCK RESPONSE] Processed prompt: '{prompt[:60]}...' using model {model_id}.",
                "input_tokens": len(prompt.split()),
                "output_tokens": 42,
                "status": "success (mock)",
            }

        try:
            # Modern Bedrock Converse API standardizes across Anthropic, Meta, Mistral, and Amazon
            response = self.client.converse(
                modelId=model_id,
                messages=[
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                system=[{"text": system_prompt}],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                }
            )

            output_text = response["output"]["message"]["content"][0]["text"]
            usage = response.get("usage", {})

            return {
                "model_id": model_id,
                "text": output_text,
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "status": "success",
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]
            logger.error("AWS Bedrock ClientError [%s]: %s", error_code, error_msg)
            raise RuntimeError(f"AWS Bedrock error ({error_code}): {error_msg}") from e
        except Exception as e:
            logger.error("Unexpected error invoking Bedrock: %s", e)
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant on AWS.",
        model_id: str = MODEL_CLAUDE_3_5_SONNET,
        max_tokens: int = 1000,
        temperature: float = 0.5,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens back in real-time using invoke_model_with_response_stream or ConverseStream.
        """
        if self.is_mock:
            mock_tokens = [
                "Hello! ", "This ", "is ", "a ", "streamed ", "response ",
                "simulating ", "Amazon ", "Bedrock ", "Claude ", "3.5 ", "Sonnet ", "on ", "AWS."
            ]
            for token in mock_tokens:
                yield token
            return

        try:
            response = self.client.converse_stream(
                modelId=model_id,
                messages=[
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                system=[{"text": system_prompt}],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                }
            )

            stream = response.get("stream")
            if stream:
                for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta:
                            yield delta["text"]

        except Exception as e:
            logger.error("Error during Bedrock streaming: %s", e)
            yield f"\n[Stream Error: {str(e)}]"

    def generate_embeddings(
        self,
        texts: List[str],
        model_id: str = MODEL_TITAN_EMBEDDING,
    ) -> List[List[float]]:
        """Generate dense text embeddings using Amazon Titan Embeddings."""
        if self.is_mock:
            # Return dummy 1536-dimensional mock vector
            return [[0.01 * (i % 100) for i in range(1536)] for _ in texts]

        embeddings = []
        for text in texts:
            body = json.dumps({"inputText": text})
            response = self.client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            response_body = json.loads(response["body"].read())
            embeddings.append(response_body.get("embedding", []))

        return embeddings
