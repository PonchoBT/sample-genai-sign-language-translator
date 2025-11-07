import json
import os
import boto3
import time
import logging
import sys
from pathlib import Path

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AgentCore configuration
AGENTCORE_AGENT_ID = os.environ.get('AGENTCORE_AGENT_ID')
AGENTCORE_AGENT_ARN = os.environ.get('AGENTCORE_AGENT_ARN')
AGENTCORE_REGION = os.environ.get('AGENTCORE_REGION', 'us-west-2')

# Initialize Bedrock AgentCore client
bedrock_agentcore = boto3.client('bedrock-agentcore-runtime', region_name=AGENTCORE_REGION)

AGENT_AVAILABLE = bool(AGENTCORE_AGENT_ID and AGENTCORE_AGENT_ARN)
if AGENT_AVAILABLE:
    logger.info(f"AgentCore agent available: {AGENTCORE_AGENT_ID}")
else:
    logger.warning("AgentCore agent not configured")

def lambda_handler(event, context):
    """
    REST API handler that routes requests to the Strands agent instead of Step Functions
    Maintains backward compatibility with existing API response format
    """
    print('received event:')
    print(event)
    
    try:
        # Handle CORS preflight requests
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Headers': '*',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
                }
            }
        
        # Check if agent is available
        if not AGENT_AVAILABLE:
            return format_error_response("AgentCore agent is not available", 503)
        
        # Extract query parameters
        query_params = event.get("queryStringParameters") or {}
        
        # Handle status check requests (backward compatibility)
        if "sfn_execution_arn" in query_params:
            return handle_legacy_status_request(query_params["sfn_execution_arn"])
        
        # Build input text for the agent
        input_text, metadata = build_agent_input(query_params, event)
        
        # Generate session ID
        session_id = event.get("requestContext", {}).get("requestId", str(int(time.time())))
        
        logger.info(f"Invoking AgentCore agent {AGENTCORE_AGENT_ID} with input: {input_text}")
        
        # Invoke the AgentCore agent
        response = bedrock_agentcore.invoke_agent(
            agentId=AGENTCORE_AGENT_ID,
            sessionId=session_id,
            inputText=input_text
        )
        
        # Process the streaming response
        agent_response = process_agentcore_response(response)
        
        # Format response to maintain API compatibility
        formatted_response = format_agent_response(agent_response, query_params)
        
        logger.info("AgentCore agent invocation completed successfully")
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(formatted_response)
        }
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return format_error_response(error_msg, 500)

def build_agent_input(query_params, event):
    """Build agent input text from API request parameters"""
    metadata = {}
    
    # Determine request type and build appropriate input
    if "Gloss" in query_params:
        # Direct gloss-to-video request
        input_text = f"Convert this ASL gloss to video: {query_params['Gloss']}"
        metadata["gloss"] = query_params["Gloss"]
        metadata["type"] = "gloss"
        
    elif "Text" in query_params:
        # Text-to-ASL translation request
        input_text = query_params["Text"]
        metadata["text"] = query_params["Text"]
        metadata["type"] = "text"
        
    elif "BucketName" in query_params and "KeyName" in query_params:
        # Audio-to-ASL translation request
        input_text = f"Process audio file from S3 bucket {query_params['BucketName']} with key {query_params['KeyName']} and convert to ASL"
        metadata["bucket_name"] = query_params["BucketName"]
        metadata["key_name"] = query_params["KeyName"]
        metadata["type"] = "audio"
        
    else:
        # Default to text processing if no specific parameters
        input_text = query_params.get("message", "Hello")
        metadata["type"] = "text"
    
    return input_text, metadata

def process_agentcore_response(response):
    """Process AgentCore streaming response and extract the result"""
    try:
        # AgentCore returns a streaming response
        event_stream = response.get('completion', [])
        
        result_text = ""
        result_data = {}
        
        for event in event_stream:
            if 'chunk' in event:
                chunk = event['chunk']
                if 'bytes' in chunk:
                    # Decode the bytes to text
                    chunk_text = chunk['bytes'].decode('utf-8')
                    result_text += chunk_text
            elif 'trace' in event:
                # Handle trace events for debugging
                trace = event['trace']
                logger.info(f"Agent trace: {trace}")
            elif 'returnControl' in event:
                # Handle return control events
                return_control = event['returnControl']
                logger.info(f"Agent return control: {return_control}")
                result_data = return_control
        
        # Try to parse as JSON if possible
        try:
            parsed_result = json.loads(result_text)
            return parsed_result
        except json.JSONDecodeError:
            # Return as plain text if not JSON
            return result_text if result_text else result_data
            
    except Exception as e:
        logger.error(f"Error processing AgentCore response: {e}", exc_info=True)
        return f"Error processing agent response: {str(e)}"

def format_agent_response(agent_response, query_params):
    """Format agent response using the enhanced response formatter"""
    try:
        # Import the response formatter
        from response_formatters import format_rest_api_response
        
        # Use the enhanced formatter for better compatibility and debugging
        formatted_response = format_rest_api_response(
            agent_response=agent_response,
            request_params=query_params,
            include_debug=os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
        )
        
        return formatted_response
        
    except ImportError:
        logger.warning("Response formatter not available, using fallback formatting")
        return format_agent_response_fallback(agent_response, query_params)
    except Exception as e:
        logger.warning(f"Error using enhanced formatter: {e}, falling back to basic formatting")
        return format_agent_response_fallback(agent_response, query_params)

def format_agent_response_fallback(agent_response, query_params):
    """Fallback formatting method for backward compatibility"""
    try:
        # Try to parse agent response as JSON if it contains structured data
        if isinstance(agent_response, str):
            # Look for JSON-like content in the response
            import re
            json_match = re.search(r'\{[^{}]*"[^"]*URL"[^{}]*\}', agent_response)
            if json_match:
                try:
                    structured_data = json.loads(json_match.group())
                    return structured_data
                except json.JSONDecodeError:
                    pass
        
        # For backward compatibility, try to extract key information
        response_data = {}
        
        # Extract URLs from response text
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, agent_response)
        
        # Map URLs to expected format
        if urls:
            for i, url in enumerate(urls):
                if 'pose' in url.lower():
                    response_data['PoseURL'] = url
                elif 'sign' in url.lower():
                    response_data['SignURL'] = url
                elif 'avatar' in url.lower():
                    response_data['AvatarURL'] = url
                elif i == 0 and 'PoseURL' not in response_data:
                    response_data['PoseURL'] = url
                elif i == 1 and 'SignURL' not in response_data:
                    response_data['SignURL'] = url
                elif i == 2 and 'AvatarURL' not in response_data:
                    response_data['AvatarURL'] = url
        
        # Extract gloss information
        gloss_match = re.search(r'(?:ASL Gloss|Gloss):\s*([^\n]+)', agent_response, re.IGNORECASE)
        if gloss_match:
            response_data['Gloss'] = gloss_match.group(1).strip()
        elif "Gloss" in query_params:
            response_data['Gloss'] = query_params["Gloss"]
        
        # Extract text information
        text_match = re.search(r'(?:Original text|Text):\s*"([^"]+)"', agent_response, re.IGNORECASE)
        if text_match:
            response_data['Text'] = text_match.group(1).strip()
        elif "Text" in query_params:
            response_data['Text'] = query_params["Text"]
        
        # If no structured data found, return the response as-is with metadata
        if not response_data:
            response_data = {
                "message": agent_response,
                "status": "completed",
                "timestamp": int(time.time())
            }
        
        return response_data
        
    except Exception as e:
        logger.warning(f"Error formatting agent response: {e}")
        return {
            "message": str(agent_response),
            "status": "completed",
            "timestamp": int(time.time())
        }

def handle_legacy_status_request(execution_arn):
    """Handle legacy Step Functions status requests for backward compatibility"""
    logger.info(f"Handling legacy status request for: {execution_arn}")
    
    # For backward compatibility, return a completed status
    # In a real migration, you might want to track agent execution status
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps({
            "status": "SUCCEEDED",
            "message": "Request processed by AgentCore agent",
            "timestamp": int(time.time())
        })
    }

def format_error_response(error_message, status_code=500):
    """Format error response with proper CORS headers"""
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps({
            "error": error_message,
            "status": "failed",
            "timestamp": int(time.time())
        })
    }

