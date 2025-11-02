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

# Add signlanguageagent module path for agent imports
current_dir = Path(__file__).parent
functions_dir = current_dir.parent
agent_path = functions_dir / 'signlanguageagent'
if agent_path.exists() and str(agent_path) not in sys.path:
    sys.path.insert(0, str(agent_path))

# Import the Strands agent
try:
    from slagent import app as agent_app
    AGENT_AVAILABLE = True
    logger.info("Successfully imported Strands agent")
except ImportError as e:
    logger.error(f"Failed to import Strands agent: {e}")
    AGENT_AVAILABLE = False

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
            return format_error_response("Strands agent is not available", 503)
        
        # Extract query parameters
        query_params = event.get("queryStringParameters") or {}
        
        # Handle status check requests (backward compatibility)
        if "sfn_execution_arn" in query_params:
            return handle_legacy_status_request(query_params["sfn_execution_arn"])
        
        # Route request to appropriate agent workflow
        agent_payload = build_agent_payload(query_params, event)
        
        logger.info(f"Invoking agent with payload: {agent_payload}")
        
        # Invoke the Strands agent
        agent_response = agent_app.invoke(agent_payload)
        
        # Format response to maintain API compatibility
        formatted_response = format_agent_response(agent_response, query_params)
        
        logger.info("Agent invocation completed successfully")
        
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

def build_agent_payload(query_params, event):
    """Build agent payload from API request parameters"""
    payload = {
        "session_id": event.get("requestContext", {}).get("requestId"),
        "metadata": {}
    }
    
    # Determine request type and build appropriate payload
    if "Gloss" in query_params:
        # Direct gloss-to-video request
        payload["message"] = f"Convert this ASL gloss to video: {query_params['Gloss']}"
        payload["type"] = "text"
        payload["metadata"]["gloss"] = query_params["Gloss"]
        
    elif "Text" in query_params:
        # Text-to-ASL translation request
        payload["message"] = query_params["Text"]
        payload["type"] = "text"
        payload["metadata"]["text"] = query_params["Text"]
        
    elif "BucketName" in query_params and "KeyName" in query_params:
        # Audio-to-ASL translation request
        payload["message"] = f"Process audio file from S3 and convert to ASL"
        payload["type"] = "audio"
        payload["metadata"]["bucket_name"] = query_params["BucketName"]
        payload["metadata"]["key_name"] = query_params["KeyName"]
        payload["BucketName"] = query_params["BucketName"]
        payload["KeyName"] = query_params["KeyName"]
        
    else:
        # Default to text processing if no specific parameters
        message = query_params.get("message", "Hello")
        payload["message"] = message
        payload["type"] = "text"
    
    return payload

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
            "message": "Request processed by Strands agent",
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

