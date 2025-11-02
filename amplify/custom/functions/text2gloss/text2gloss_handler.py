import json
import os
import time
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from strands import tool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

eng_to_asl_model = os.environ.get('ENG_TO_ASL_MODEL', 'amazon.nova-lite-v1:0')


def construct_query(text: str) -> str:
    """Construct the prompt for text-to-gloss conversion"""
    return f"""

H: Here are some examples of translations from english text to ASL gloss 
Examples:
Apples ==> APPLE
you  ==> IX-2P
your  ==> IX-2P
Love ==> LIKE
My ==> IX-1P
Thanks ==> THANK-YOU
am ==> 
and ==> 
be ==>
of ==>
video ==> MOVIE
image ==> PICTURE
conversations ==> TALK
type of ==> TYPE
Watch ==> SEE

Translate the following english text to ASL Gloss and return only the gloss. Don't provide any explanation.
{text} ==>


A:"""


def lambda_handler(event, context):
    """Invoke Bedrock to convert English text to ASL Gloss
    Parameters
    ----------
    event: dict, required
        Input event to the Lambda function
    context: object, required
        Lambda Context runtime methods and attributes
    Returns
    ------
        dict: text consists of ASL Gloss
    """
    try:
        input_text = event.get("Text", "")
        if not input_text.strip():
            return {'Error': 'No text provided for translation'}
        
        gloss = text_to_asl_gloss(input_text)
        return {'Gloss': gloss, 'Text': input_text}
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {'Error': f'Translation failed: {str(e)}'}


@tool
def text_to_asl_gloss(text: str) -> str:
    """Convert English text to ASL gloss notation with error handling and retry logic
    
    Args:
        text: English text to convert to ASL gloss
        
    Returns:
        str: ASL gloss notation
        
    Raises:
        ValueError: If text is empty or invalid
        RuntimeError: If translation fails after retries
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    text = text.strip()
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            bedrock_client = boto3.client(service_name="bedrock-runtime")
            
            # Create the prompt
            prompt_data = construct_query(text)
            conversation = [
                {
                    "role": "user",
                    "content": [{"text": prompt_data}],
                }
            ]
            
            inference_config = {
                "maxTokens": 3000, 
                "temperature": 0.0, 
                "topP": 0.5
            }

            logger.info(f"Attempting text-to-gloss conversion (attempt {attempt + 1}/{max_retries})")
            
            response = bedrock_client.converse(
                modelId=eng_to_asl_model,
                messages=conversation,
                inferenceConfig=inference_config,
            )

            gloss = response["output"]["message"]["content"][0]["text"]
            
            # Clean up the gloss output
            gloss = gloss.strip()
            if gloss.startswith("==>"):
                gloss = gloss[3:].strip()
            
            logger.info(f"Successfully converted text to gloss: '{text}' -> '{gloss}'")
            return gloss
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', 'Unknown')
            
            logger.warning(f"Bedrock ClientError on attempt {attempt + 1}: {error_code} - {error_message}")
            
            # Don't retry on certain error types
            if error_code in ['ValidationException', 'AccessDeniedException']:
                raise RuntimeError(f"Non-retryable error: {error_code} - {error_message}")
            
            if attempt == max_retries - 1:
                raise RuntimeError(f"Text-to-gloss conversion failed after {max_retries} attempts: {error_message}")
                
        except BotoCoreError as e:
            logger.warning(f"BotoCoreError on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Text-to-gloss conversion failed after {max_retries} attempts: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Text-to-gloss conversion failed after {max_retries} attempts: {str(e)}")
        
        # Exponential backoff with jitter
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + (time.time() % 1)  # Add jitter
            logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "what is your name?",
        "How are you?", 
        "She is watching a movie",
        "He wants to play",
        "Can you come with me?"
    ]
    
    for test_text in test_cases:
        try:
            result = lambda_handler({"Text": test_text}, {})
            print(f"Input: {test_text}")
            print(f"Result: {result}")
            print("-" * 50)
        except Exception as e:
            print(f"Error testing '{test_text}': {str(e)}")