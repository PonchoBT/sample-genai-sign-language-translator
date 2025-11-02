"""
GenASL Sign Language Agent - Strands-based agent for ASL translation
This module implements the main agent entry point and configuration for the
GenASL system using AWS Bedrock AgentCore and Strands framework.
"""

import json
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Import utility functions
from .utils import setup_logging, retry_with_backoff, validate_payload, extract_response_content
from .config import config

# Configure logging
logger = setup_logging(config.agent.log_level)

# Import the AgentCore SDK
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent



# Set up module paths for tool imports
def setup_module_paths():
    """Set up Python paths to import tools from sibling directories"""
    current_dir = Path(__file__).parent
    functions_dir = current_dir.parent
    
    # Add text2gloss module path
    text2gloss_path = functions_dir / 'text2gloss'
    if text2gloss_path.exists():
        sys.path.insert(0, str(text2gloss_path))
        logger.info(f"Added text2gloss path: {text2gloss_path}")
    
    # Add gloss2pose module path
    gloss2pose_path = functions_dir / 'gloss2pose'
    if gloss2pose_path.exists():
        sys.path.insert(0, str(gloss2pose_path))
        logger.info(f"Added gloss2pose path: {gloss2pose_path}")

# Set up paths and import tools
setup_module_paths()

try:
    from text2gloss_handler import text_to_asl_gloss
    from gloss2pose_handler import gloss_to_video
    logger.info("Successfully imported text2gloss and gloss2pose tools")
except ImportError as e:
    logger.error(f"Failed to import tools: {e}")
    # Define placeholder tools for development/testing
    from strands import tool
    
    @tool
    def text_to_asl_gloss(text: str) -> str:
        """Placeholder text to ASL gloss conversion tool"""
        logger.warning("Using placeholder text_to_asl_gloss tool")
        return f"PLACEHOLDER_GLOSS_FOR_{text.upper().replace(' ', '_')}"
    
    @tool
    def gloss_to_video(gloss_sentence: str, text: str = None, pose_only: bool = False, pre_sign: bool = True) -> dict:
        """Placeholder gloss to video conversion tool"""
        logger.warning("Using placeholder gloss_to_video tool")
        return {
            'PoseURL': 'https://placeholder-pose-url.com',
            'SignURL': 'https://placeholder-sign-url.com',
            'AvatarURL': 'https://placeholder-avatar-url.com',
            'Gloss': gloss_sentence,
            'Text': text
        }

# System prompt for the agent
SYSTEM_PROMPT = """You are GenASL, an AI-powered American Sign Language translation agent. Your primary function is to help users translate English text and audio into American Sign Language (ASL) videos.

Your capabilities include:
1. Converting English text to ASL gloss notation
2. Generating ASL pose sequences and avatar videos from gloss
3. Processing audio input for transcription and translation
4. Providing conversational interaction and status updates

When a user provides text for translation:
1. First use the text_to_asl_gloss tool to convert the English text to ASL gloss
2. Then use the gloss_to_video tool to generate the corresponding ASL video
3. Provide the user with the video URLs and explain what was translated

Always be helpful, clear, and provide status updates during processing. If errors occur, explain them in user-friendly terms and suggest alternatives when possible.

Remember to maintain context across conversations and handle multiple translation requests appropriately."""

# Create the AgentCore app
app = BedrockAgentCoreApp()

# Initialize the Strands agent
agent = Agent(
    model=config.model.eng_to_asl_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[text_to_asl_gloss, gloss_to_video],
    max_tokens=config.model.max_tokens,
    temperature=config.model.temperature
)

logger.info(f"Agent initialized with {len(agent.tools)} tools")

@app.entrypoint
@retry_with_backoff(max_retries=config.agent.max_retries, delay=config.agent.retry_delay)
def invoke(payload: Dict[str, Any]) -> str:
    """
    Main entry point for the GenASL agent
    
    Args:
        payload: Dictionary containing the request data
                Expected keys:
                - message: The user's message/text to process
                - type: Optional request type ('text', 'audio', 'video')
                - metadata: Optional additional context
    
    Returns:
        str: The agent's response message
    """
    try:
        logger.info(f"Agent invoked with payload keys: {list(payload.keys())}")
        
        # Validate and normalize payload
        normalized_payload = validate_payload(payload)
        
        user_message = normalized_payload["message"]
        request_type = normalized_payload["type"]
        metadata = normalized_payload["metadata"]
        
        logger.info(f"Processing {request_type} request: {user_message[:100]}...")
        
        # Invoke the agent with the user message
        response = agent(user_message)
        
        # Extract the response content using utility function
        result = extract_response_content(response)
        
        logger.info(f"Agent response generated successfully (length: {len(result)})")
        return result
        
    except ValueError as e:
        error_msg = f"Invalid request: {str(e)}"
        logger.warning(f"Validation error: {e}")
        return error_msg
        
    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        logger.error(f"Agent invocation error: {e}", exc_info=True)
        return error_msg

def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring"""
    try:
        return {
            "status": "healthy",
            "agent_model": config.model.eng_to_asl_model,
            "tools_count": len(agent.tools),
            "config": {
                "pose_bucket": config.aws.pose_bucket,
                "data_bucket": config.aws.asl_data_bucket,
                "table_name": config.aws.table_name,
                "region": config.aws.region
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info("Starting GenASL Sign Language Agent")
    logger.info(f"Health check: {health_check()}")
    
    # Run the AgentCore app
    app.run()