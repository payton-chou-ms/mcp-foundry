import os
import json
import logging
import sys
import requests
from mcp_foundry.mcp_server import mcp
from mcp.server.fastmcp import Context
from dotenv import load_dotenv

# Configure logging (following the pattern from other tools)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_foundry_finetuning")

load_dotenv()

# Use consistent environment variable names following the codebase pattern
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")

@mcp.tool()
def list_finetuning_jobs(ctx: Context):
    """
    MCP-compatible function to list all finetuning jobs using Azure OpenAI API.
    
    Returns:
        List of dictionaries containing job ID and status.
    """

    if not azure_endpoint or not api_key:
        return json.dumps({
            "error": "Missing required environment variables: 'AZURE_OPENAI_ENDPOINT' or 'AZURE_OPENAI_API_KEY'."
        })

    url = f"{azure_endpoint}/openai/fine_tuning/jobs?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        jobs_data = response.json()
        return [{"job_id": job["id"], "status": job["status"]} for job in jobs_data.get("data", [])]
    else:
        print(f"Failed to retrieve jobs. Status code: {response.status_code}")
        return []
    
@mcp.tool()
def get_finetuning_job_events(ctx: Context, job_id: str):
    """
    MCP-compatible function to retrieve all events for a specific finetuning job.
    It also returns the billing details.

    Returns:
        List of event details including timestamp and message.
    """

    if not azure_endpoint or not api_key:
        return json.dumps({
            "error": "Missing required environment variables: 'AZURE_OPENAI_ENDPOINT' or 'AZURE_OPENAI_API_KEY'."
        })
    
    url = f"{azure_endpoint}/openai/fine_tuning/jobs/{job_id}/events?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        events_data = response.json()
        return [
            {
                "timestamp": event.get("created_at"),
                "message": event.get("message")
            }
            for event in events_data.get("data", [])
        ]
    else:
        print(f"Failed to retrieve events. Status code: {response.status_code}")
        return []


@mcp.tool()
def fetch_finetuning_status(ctx: Context, job_id: str) -> str:
    """
    Fetches the status of a fine-tuning job using Azure OpenAI API.

    Parameters:
    - job_id: The ID of the fine-tuning job.

    Returns:
    - Job status information as a JSON string.
    """

    if not azure_endpoint or not api_key:
        return json.dumps({
            "error": "Missing required environment variables: 'AZURE_OPENAI_ENDPOINT' or 'AZURE_OPENAI_API_KEY'."
        })

    url = f"{azure_endpoint}/openai/fine_tuning/jobs/{job_id}?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"Fetching fine-tuning job status for job_id: {job_id}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        job_data = response.json()

        return json.dumps({
            "job_id": job_data.get("id"),
            "status": job_data.get("status"),
            "model": job_data.get("model"),
            "created_at": job_data.get("created_at"),
            "finished_at": job_data.get("finished_at"),
            "hyperparameters": job_data.get("hyperparameters"),
            "fine_tuned_model": job_data.get("fine_tuned_model"),
            "trained_tokens": job_data.get("trained_tokens"),
            "result_files": job_data.get("result_files", []),
            "training_files": job_data.get("training_files", []),
            "validation_files": job_data.get("validation_files", []),
            "estimated_finish": job_data.get("estimated_finish"),
            "error": job_data.get("error", {})
        })

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching fine-tuning job status: {str(e)}")
        return json.dumps({"error": f"Request error: {str(e)}"})
    except Exception as e:
        logger.error(f"Unexpected error fetching fine-tuning job status: {str(e)}")
        return json.dumps({"error": f"Unexpected error: {str(e)}"})

@mcp.tool()    
def get_finetuning_metrics(ctx, job_id: str) -> str:
    """
    Retrieves fine-tuning metrics if the job has succeeded.
    Calls fetch_finetuning_status to confirm job completion.
    Then fetches the result.csv content using the result_file_id.
    """
    job_data = fetch_finetuning_status(ctx, job_id)
    # Parse the JSON string to a dictionary
    job_data = json.loads(job_data)
    status = job_data.get("status")
    if status == "succeeded":
        result_files = job_data.get("result_files", [])
        file_id = result_files[0] if result_files else None

        file_url = f"{azure_endpoint}/openai/files/{file_id}/content?api-version={api_version}"
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"Fetching result file content for file_id: {file_id}")
            file_response = requests.get(file_url, headers=headers, timeout=10)
            file_response.raise_for_status()
            result_csv_content = file_response.content.decode('utf-8')

            return json.dumps({
                "job_id": job_data.get("id"),
                "status": status,
                "result_file_id": file_id,
                "result_file_url": file_url,
                "result_csv_content": result_csv_content
            })
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching result file: {str(e)}")
            return json.dumps({"error": f"Request error fetching result file: {str(e)}"})
        except Exception as e:
            logger.error(f"Unexpected error fetching result file: {str(e)}")
            return json.dumps({"error": f"Unexpected error fetching result file: {str(e)}"})
    else:
        return json.dumps({
            "job_id": job_data.get("id"),
            "status": status,
            "message": "Job has not succeeded yet."
        })

# Problem statement:
# We need to validate the file format for finetuning jobs and ensure compatibility with the job

@mcp.tool()
def validate_and_create_finetuning_job(
    ctx: Context, 
    job_type: str, 
    training_file_id: str,
    validation_file_id: str = None,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict = None
) -> str:
    """
    Validates file compatibility for finetuning jobs and creates the job if valid.
    
    Parameters:
    - job_type: Type of finetuning job ('supervised' or 'dpo')
    - training_file_id: ID of the training file
    - validation_file_id: Optional ID of the validation file
    - model: Base model to fine-tune (default: gpt-4o-mini-2024-07-18)
    - hyperparameters: Optional hyperparameters for the job
    
    Returns:
    - JSON string with job creation result or validation error
    """
    
    if not azure_endpoint or not api_key:
        return json.dumps({
            "error": "Missing required environment variables: 'AZURE_OPENAI_ENDPOINT' or 'AZURE_OPENAI_API_KEY'."
        })
    
    # Validate job type
    if job_type not in ['supervised', 'dpo']:
        return json.dumps({
            "error": "Invalid job_type. Must be 'supervised' or 'dpo'."
        })
    
    # First, check the file content to validate format
    file_validation_result = validate_finetuning_file_format(training_file_id, job_type)
    
    if not file_validation_result['is_valid']:
        # File format doesn't match the job type
        return json.dumps({
            "error": "File format incompatibility detected",
            "details": file_validation_result['message'],
            "detected_format": file_validation_result['detected_format'],
            "expected_format": file_validation_result['expected_format'],
            "resolution_options": [
                f"Change job type to '{file_validation_result['detected_format']}'",
                f"Provide a file with '{file_validation_result['expected_format']}' format"
            ],
            "action_required": "Please use 'resolve_finetuning_compatibility' tool to proceed"
        })
    
    # If validation passes, create the finetuning job
    return _create_finetuning_job(
        job_type=job_type,
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        model=model,
        hyperparameters=hyperparameters
    )


def validate_finetuning_file_format(file_id: str, expected_job_type: str) -> dict:
    """
    Validates the format of a JSONL file against the expected fine-tuning job type.

    Parameters:
    - file_id: ID of the file to validate.
    - expected_job_type: Expected format type ('supervised' or 'dpo').

    Returns:
    - dict containing:
        - is_valid (bool): Whether the file matches the expected format.
        - message (str): Explanation of the validation result.
        - detected_format (str): Detected format ('supervised', 'dpo', or 'unknown').
        - expected_format (str): The expected format provided as input.
    """

    # Construct URL to fetch file content from Azure OpenAI
    file_url = f"{azure_endpoint}/openai/files/{file_id}/content?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    try:
        # Fetch file content
        response = requests.get(file_url, headers=headers, timeout=10)
        response.raise_for_status()
        file_content = response.text

        # Split file content into individual lines (JSONL format)
        lines = file_content.strip().split('\n')

        # Check if file is empty
        if not lines or all(not line.strip() for line in lines):
            return {
                'is_valid': False,
                'message': 'File is empty or contains only whitespace.',
                'detected_format': 'unknown',
                'expected_format': expected_job_type
            }

        max_lines_to_check = min(10, len(lines))  # Check up to 10 lines for robust detection
        dpo_flag = False
        supervised_flag = False
        total_checked = 0

        # Iterate through lines to detect format
        for line in lines[:max_lines_to_check]:
            if not line.strip():  # Skip empty lines
                continue
                
            total_checked += 1
            try:
                data = json.loads(line)
                logger.info(f"DaTA FILE {data}")

                # Check for DPO format with two possible structures:
                # 1. Classic DPO: {'prompt', 'chosen', 'rejected'}
                # 2. New DPO: {'input': {...}, 'preferred_output': [...], 'non_preferred_output': [...]}
                
                # Classic DPO format and new DPO format
                # ...existing code...
                # Classic DPO format
                if all(key in data for key in ['prompt', 'chosen', 'rejected']):
                    dpo_flag = True
                    break

                # New DPO format
                if all(key in data for key in ['input', 'preferred_output', 'non_preferred_output']):
                    dpo_flag = True
                    break

                # Check for supervised format: must contain 'messages' as a list of dicts with 'role' and 'content'
                if ('messages' in data and isinstance(data['messages'], list) and
                    len(data['messages']) > 0 and
                    all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in data['messages'])):
                    supervised_flag = True
                    continue

            except json.JSONDecodeError as e:
                # Log JSON decoding errors and continue checking other lines
                logger.warning(f"JSON decode error in line {total_checked}: {str(e)[:100]}")
                continue

        # Determine detected format based on counts
        if dpo_flag:
            detected_format = 'dpo'
        elif supervised_flag:
            detected_format = 'supervised'
        else:
            detected_format = 'unknown'

        # Check if detected format matches expected format
        is_valid = detected_format == expected_job_type

        # Construct informative validation message
        if detected_format == 'unknown':
            message = (f"Unable to detect file format after checking {total_checked} lines. "
                      f"Ensure the file matches 'supervised' or 'dpo' JSONL structure.")
        elif is_valid:
            message = f"File format successfully validated as '{expected_job_type}' format."
        else:
            message = (f"File format mismatch: Expected '{expected_job_type}' but detected '{detected_format}'. "
                      f"Please provide a file matching the '{expected_job_type}' format or change the job type.")

        # Return validation result with additional debug info
        return {
            'is_valid': is_valid,
            'message': message,
            'detected_format': detected_format,
            'expected_format': expected_job_type,
            'lines_checked': total_checked,
            'supervised_flag': supervised_flag,
            'dpo_flag': dpo_flag
        }

    except requests.exceptions.RequestException as e:
        # Handle request-related errors
        logger.error(f"Request error fetching file content: {str(e)}")
        return {
            'is_valid': False,
            'message': f"Request error fetching file content: {str(e)}",
            'detected_format': 'unknown',
            'expected_format': expected_job_type
        }
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error validating file format: {str(e)}")
        return {
            'is_valid': False,
            'message': f"Unexpected error: {str(e)}",
            'detected_format': 'unknown',
            'expected_format': expected_job_type
        }

@mcp.tool()
def resolve_finetuning_compatibility(
    ctx: Context,
    resolution_choice: str,
    original_job_type: str,
    training_file_id: str,
    validation_file_id: str = None,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict = None,
    new_file_id: str = None
) -> str:
    """
    Resolves file compatibility issues and creates the finetuning job.
    
    Parameters:
    - resolution_choice: Either 'change_job_type' or 'change_file'
    - original_job_type: The originally requested job type
    - training_file_id: Original training file ID
    - validation_file_id: Optional validation file ID
    - model: Base model to fine-tune
    - hyperparameters: Optional hyperparameters
    - new_file_id: New file ID if resolution_choice is 'change_file'
    
    Returns:
    - JSON string with job creation result
    """
    
    if resolution_choice not in ['change_job_type', 'change_file']:
        return json.dumps({
            "error": "Invalid resolution_choice. Must be 'change_job_type' or 'change_file'."
        })
    
    if resolution_choice == 'change_file' and not new_file_id:
        return json.dumps({
            "error": "new_file_id is required when resolution_choice is 'change_file'."
        })
    
    # Determine the actual job type and file to use
    if resolution_choice == 'change_job_type':
        # Detect the file format and use that as job type
        file_validation = validate_finetuning_file_format(training_file_id, original_job_type)
        actual_job_type = file_validation['detected_format']
        actual_training_file_id = training_file_id
    else:  # change_file
        actual_job_type = original_job_type
        actual_training_file_id = new_file_id
        
        # Validate the new file
        file_validation = validate_finetuning_file_format(new_file_id, original_job_type)
        if not file_validation['is_valid']:
            return json.dumps({
                "error": "New file still has incompatible format",
                "details": file_validation['message']
            })
    
    # Create the finetuning job
    return _create_finetuning_job(
        job_type=actual_job_type,
        training_file_id=actual_training_file_id,
        validation_file_id=validation_file_id,
        model=model,
        hyperparameters=hyperparameters
    )

def _create_finetuning_job(
    job_type: str,
    training_file_id: str,
    validation_file_id: str = None,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict = None
) -> str:
    """
    Internal function to create a finetuning job.
    """
    
    url = f"{azure_endpoint}/openai/fine_tuning/jobs?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Prepare request body
    body = {
        "model": model,
        "training_file": training_file_id,
    }
    
    if validation_file_id:
        body["validation_file"] = validation_file_id
    
    if hyperparameters:
        body["hyperparameters"] = hyperparameters
    else:
        # Default hyperparameters based on job type
        if job_type == 'dpo':
            body["hyperparameters"] = {
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 1.8
            }
        else:  # supervised
            body["hyperparameters"] = {
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 2
            }
    
    # Add method for DPO jobs
    if job_type == 'dpo':
        # Support custom DPO parameters if provided in hyperparameters['dpo']
        method = {"type": "dpo"}
        if hyperparameters and "dpo" in hyperparameters:
            method.update(hyperparameters["dpo"])
        body["method"] = method
    
    try:
        logger.info(f"Creating {job_type} finetuning job with file: {training_file_id}")
        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        
        job_data = response.json()
        return json.dumps({
            "success": True,
            "job_id": job_data.get("id"),
            "status": job_data.get("status"),
            "job_type": job_type,
            "model": job_data.get("model"),
            "training_file": job_data.get("training_file"),
            "created_at": job_data.get("created_at"),
            "message": f"{job_type.upper()} finetuning job created successfully"
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error creating finetuning job: {str(e)}")
        return json.dumps({
            "success": False,
            "error": f"Request error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Unexpected error creating finetuning job: {str(e)}")
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        })

# end of problem statement code