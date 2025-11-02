import multiprocessing
import os
import re
import subprocess
import sys
import time
import logging
from threading import Thread
from typing import Optional, Dict, Any, List

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError, BotoCoreError
import uuid
import pathlib
from strands import tool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables with defaults
pose_bucket = os.environ.get('POSE_BUCKET', '')
asl_data_bucket = os.environ.get('ASL_DATA_BUCKET', '')
key_prefix = os.environ.get("KEY_PREFIX", "")
table_name = os.environ.get('TABLE_NAME', '')
output_ext = 'webm'


def lambda_handler(event, context):
    """Lambda handler for gloss-to-video conversion
    
    Parameters
    ----------
    event: dict, required
        Input event to the Lambda function
    context: object, required
        Lambda Context runtime methods and attributes
    Returns
    ------
        dict: Object containing video URLs and metadata
    """
    try:
        gloss = event.get("Gloss", "")
        text = event.get('Text')
        
        if not gloss.strip():
            return {'Error': 'No gloss provided for video generation'}
        
        result = gloss_to_video(gloss, text)
        return result
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {'Error': f'Video generation failed: {str(e)}'}


@tool
def gloss_to_video(gloss_sentence: str, text: Optional[str] = None, 
                  pose_only: bool = False, pre_sign: bool = True) -> Dict[str, Any]:
    """Convert ASL gloss to pose sequences and generate videos with error handling
    
    Args:
        gloss_sentence: ASL gloss string to convert
        text: Optional original text for reference
        pose_only: If True, only generate pose video
        pre_sign: If True, generate presigned URLs for S3 objects
        
    Returns:
        dict: Dictionary containing video URLs and metadata
        
    Raises:
        ValueError: If gloss_sentence is empty or invalid
        RuntimeError: If video generation fails
    """
    if not gloss_sentence or not gloss_sentence.strip():
        raise ValueError("Gloss sentence cannot be empty")
    
    # Validate environment variables
    required_env_vars = {
        'POSE_BUCKET': pose_bucket,
        'ASL_DATA_BUCKET': asl_data_bucket,
        'TABLE_NAME': table_name,
        'KEY_PREFIX': key_prefix
    }
    
    missing_vars = [var for var, value in required_env_vars.items() if not value]
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    gloss_sentence = gloss_sentence.strip()
    uniq_key = str(uuid.uuid4())
    
    logger.info(f"Starting gloss-to-video conversion for: '{gloss_sentence}' (ID: {uniq_key})")
    
    try:
        # Get sign IDs from DynamoDB with error handling
        sign_ids = get_sign_ids_from_gloss(gloss_sentence)
        
        if not sign_ids:
            logger.warning(f"No sign IDs found for gloss: '{gloss_sentence}'")
            return {
                'Error': f'No signs found for gloss: {gloss_sentence}',
                'Gloss': gloss_sentence,
                'Text': text
            }
        
        logger.info(f"Found {len(sign_ids)} sign IDs: {sign_ids}")
        
        # Process videos with error handling
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        threads = []
        
        # Always create pose video
        pose_thread = Thread(
            target=process_videos_with_error_handling,
            args=(return_dict, "pose", sign_ids, uniq_key, pre_sign)
        )
        threads.append(pose_thread)
        pose_thread.start()
        
        # Create sign and avatar videos if not pose_only
        if not pose_only:
            sign_thread = Thread(
                target=process_videos_with_error_handling,
                args=(return_dict, "sign", sign_ids, uniq_key, pre_sign)
            )
            avatar_thread = Thread(
                target=process_videos_with_error_handling,
                args=(return_dict, "avatar", sign_ids, uniq_key, pre_sign)
            )
            threads.extend([sign_thread, avatar_thread])
            sign_thread.start()
            avatar_thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for errors in thread results
        errors = [return_dict.get(f"{video_type}_error") for video_type in ["pose", "sign", "avatar"] 
                 if return_dict.get(f"{video_type}_error")]
        
        if errors:
            error_msg = f"Video processing errors: {'; '.join(errors)}"
            logger.error(error_msg)
            return {
                'Error': error_msg,
                'Gloss': gloss_sentence,
                'Text': text
            }
        
        # Build result dictionary
        result = {
            'PoseURL': return_dict.get("pose"),
            'Gloss': gloss_sentence,
            'Text': text
        }
        
        if not pose_only:
            result.update({
                'SignURL': return_dict.get("sign"),
                'AvatarURL': return_dict.get("avatar")
            })
        
        logger.info(f"Successfully generated videos for gloss: '{gloss_sentence}'")
        return result
        
    except Exception as e:
        logger.error(f"Error in gloss_to_video: {str(e)}")
        raise RuntimeError(f"Video generation failed: {str(e)}")


def get_sign_ids_from_gloss(gloss_sentence: str) -> List[str]:
    """Extract sign IDs from gloss sentence using DynamoDB lookup with error handling"""
    sign_ids = []
    max_retries = 3
    base_delay = 1.0
    
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(table_name)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to DynamoDB table '{table_name}': {str(e)}")
    
    for gloss in gloss_sentence.split(" "):
        gloss = re.sub('[,!?.]', '', gloss.strip())
        if not gloss:
            continue
            
        for attempt in range(max_retries):
            try:
                response = table.query(
                    KeyConditionExpression=Key('Gloss').eq(gloss)
                )
                
                if response['Count'] == 0:
                    # If no sign found, finger spell it
                    logger.info(f"No direct sign found for '{gloss}', finger spelling...")
                    for char in gloss:
                        char_response = table.query(
                            KeyConditionExpression=Key('Gloss').eq(char)
                        )
                        if char_response['Count'] > 0:
                            sign_ids.append(char_response['Items'][0]['SignID'])
                else:
                    sign_ids.append(response['Items'][0]['SignID'])
                break  # Success, exit retry loop
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                logger.warning(f"DynamoDB error for gloss '{gloss}' (attempt {attempt + 1}): {error_code}")
                
                if attempt == max_retries - 1:
                    logger.error(f"Failed to query DynamoDB for gloss '{gloss}' after {max_retries} attempts")
                    continue  # Skip this gloss and continue with next
                
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Unexpected error querying gloss '{gloss}': {str(e)}")
                break  # Skip this gloss
    
    return sign_ids


def process_videos_with_error_handling(return_dict, video_type: str, sign_ids: List[str], 
                                     uniq_key: str, pre_sign: bool):
    """Wrapper for process_videos with error handling"""
    try:
        result = process_videos(return_dict, video_type, sign_ids, uniq_key, pre_sign)
        return result
    except Exception as e:
        error_msg = f"Error processing {video_type} video: {str(e)}"
        logger.error(error_msg)
        return_dict[f"{video_type}_error"] = error_msg


def process_videos(return_dict, video_type: str, sign_ids: List[str], uniq_key: str, pre_sign: bool):
    """Process videos with enhanced error handling and validation"""
    if not sign_ids:
        raise ValueError(f"No sign IDs provided for {video_type} video processing")
    
    max_retries = 2
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            s3 = boto3.client('s3')
            temp_folder = f"/tmp/{uniq_key}/"
            video_folder = f"{temp_folder}{video_type}/"
            
            # Create directories
            pathlib.Path(video_folder).mkdir(parents=True, exist_ok=True)
            
            # Download video files
            downloaded_files = []
            with open(f"{temp_folder}{video_type}.txt", 'w') as writer:
                for sign_id in sign_ids:
                    # Determine S3 key based on video type
                    if video_type == "sign":
                        key = f"{key_prefix}sign/sign-{sign_id}.mp4"
                    elif video_type == "pose":
                        key = f"{key_prefix}pose2/pose-{sign_id}.mp4"
                    else:  # avatar
                        key = f"{key_prefix}avatar/avatar-{sign_id}.mp4"
                    
                    local_file_name = f"{video_folder}{video_type}-{sign_id}.mp4"
                    
                    try:
                        logger.info(f"Downloading {key} to {local_file_name}")
                        s3.download_file(pose_bucket, key, local_file_name)
                        
                        # Verify file was downloaded and has content
                        if os.path.exists(local_file_name) and os.path.getsize(local_file_name) > 0:
                            writer.write(f"file '{local_file_name}'\n")
                            downloaded_files.append(local_file_name)
                        else:
                            logger.warning(f"Downloaded file {local_file_name} is empty or doesn't exist")
                            
                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                        logger.warning(f"Failed to download {key}: {error_code}")
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error downloading {key}: {str(e)}")
                        continue
            
            if not downloaded_files:
                raise RuntimeError(f"No video files were successfully downloaded for {video_type}")
            
            logger.info(f"Successfully downloaded {len(downloaded_files)} files for {video_type}")
            
            # Combine videos using FFmpeg
            output_file = f"{temp_folder}{video_type}.{output_ext}"
            ffmpeg_args = [
                "/opt/bin/ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", f"{temp_folder}{video_type}.txt",
                "-c:v", "libvpx-vp9",
                "-y",  # Overwrite output file
                output_file
            ]
            
            logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_args)}")
            
            try:
                result = subprocess.run(
                    ffmpeg_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    check=True,
                    timeout=300  # 5 minute timeout
                )
                logger.info(f"FFmpeg completed successfully for {video_type}")
                
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"FFmpeg timeout for {video_type} video processing")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
                raise RuntimeError(f"FFmpeg failed for {video_type}: {error_msg}")
            
            # Verify output file was created
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                raise RuntimeError(f"FFmpeg did not produce valid output file for {video_type}")
            
            # Upload to S3 or return local path
            if pre_sign:
                output_key = f"{uniq_key}/{video_type}.{output_ext}"
                try:
                    logger.info(f"Uploading {output_file} to s3://{asl_data_bucket}/{output_key}")
                    s3.upload_file(output_file, asl_data_bucket, output_key)
                    
                    # Generate presigned URL
                    video_url = s3.generate_presigned_url(
                        ClientMethod='get_object',
                        Params={
                            'Bucket': asl_data_bucket,
                            'Key': output_key
                        },
                        ExpiresIn=604800  # 7 days
                    )
                    
                    return_dict[video_type] = video_url
                    logger.info(f"Successfully uploaded and generated presigned URL for {video_type}")
                    return video_url
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    raise RuntimeError(f"Failed to upload {video_type} video to S3: {error_code}")
            else:
                return_dict[video_type] = output_file
                return output_file
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            logger.warning(f"Attempt {attempt + 1} failed for {video_type}: {str(e)}")
            delay = base_delay * (2 ** attempt)
            logger.info(f"Retrying {video_type} processing in {delay} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    # Test cases
    test_cases = [
        {"Gloss": "HELLO WORLD", "Text": "Hello world"},
        {"Gloss": "IX-1P LIKE MOVIE", "Text": "I like movies"},
        {"Gloss": "THANK-YOU", "Text": "Thank you"}
    ]
    
    for test_case in test_cases:
        try:
            result = lambda_handler(test_case, {})
            print(f"Input: {test_case}")
            print(f"Result: {result}")
            print("-" * 50)
        except Exception as e:
            print(f"Error testing {test_case}: {str(e)}")