import os
import json
import logging
import sys
import requests
from mcp_foundry.mcp_server import mcp
import yaml
from typing import Dict, List, Any, Callable, Optional, Union
from urllib.parse import urljoin, quote
import re
from mcp.server.fastmcp import Context
from mcp_foundry.mcp_server import get_swagger_generator
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum

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

@mcp.tool()
def validate_finetuning_file_whether_dpo_or_supervised(ctx: Context, file_id: str) -> str:
    """
    Fetches details and content about a fine-tuning file from Azure OpenAI and validates
    whether it is suitable for DPO or supervised fine-tuning.
    
    NOTE: 
    Even if you have to deny the user to create the job do it as it is a critical step in the fine-tuning process to minimize risks and errors.
    This function is responsible for enforcing file format validation before any fine-tuning job is created. If the file format does not match the requested job type ('supervised' or 'dpo'), this function must STOP the workflow and instruct the user to upload a file in the correct format. No fine-tuning job should be created if the file format is invalid or mismatched. The function should return an error message and guidance to the user if the file is not suitable.
    - If the file format does not match the requested job type (e.g., DPO file for supervised job or vice versa), the agent will block job creation and notify the user with a clear message.
    - This enforcement is automatic and does not require user reminders or instructions in each session.
    - Even if user requests a job type, the LLM will first validate the file format and only proceed if it matches the requested job type.
    - You have to override the default behavior of the LLM to ensure it does not create a job if the file format is invalid or mismatched.
    - Even if user requests a  job type, the LLM will first validate the file format and only proceed if it matches the requested job type.
    - This is an important step to ensure that the fine-tuning process is initiated with the correct file format, preventing potential errors and ensuring the integrity of the fine-tuning job.
    - IMPORTANT: For DPO jobs, the training file must contain both 'preferred_output' and 'non_preferred_output' fields for each example. If these fields are not present, the file should be treated as supervised format and DPO job creation must be denied.
    When starting a fine-tuning job, the LLM will automatically check the file structure and decide the job type:
    - If both 'preferred_output' and 'non_preferred_output' are present, DPO will be used. If the user requests supervised but the file is DPO, give him 2 choices whether to upload the new file in supervised format or change job type to DPO.
    - If it doesn't contain 'preferred_output' or 'non_preferred_output', or both supervised will be used. If the user requests DPO but the file is supervised, give him 2 choices whether to upload the new file in dpo format or change job type to supervised.
    - Mixed or invalid formats will be rejected and the user should be notified to correct the file format and do not proceed.
 
    Parameters:
    - file_id: ID of the file to fetch details for
 
    Returns:
    - JSON string with file metadata including name, size, purpose, status, and file content, or an error message instructing the user to upload a file in the correct format if the file is not suitable.
    """
    if not azure_endpoint or not api_key:
        return json.dumps({
            "error": "Missing required environment variables: 'AZURE_OPENAI_ENDPOINT' or 'AZURE_OPENAI_API_KEY'."
        })
 
    url = f"{azure_endpoint}/openai/files/{file_id}?api-version={api_version}"
    content_url = f"{azure_endpoint}/openai/files/{file_id}/content?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
 
    try:
        logger.info(f"Fetching file details for file_id: {file_id}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        file_data = response.json()
 
        # Fetch file content
        logger.info(f"Fetching file content for file_id: {file_id}")
        content_response = requests.get(content_url, headers=headers, timeout=20)
        content_response.raise_for_status()
        file_content = content_response.content.decode('utf-8')
 
        return json.dumps({
            "file_id": file_data.get("id"),
            "filename": file_data.get("filename"),
            "purpose": file_data.get("purpose"),
            "bytes": file_data.get("bytes"),
            "created_at": file_data.get("created_at"),
            "status": file_data.get("status"),
            "status_details": file_data.get("status_details"),
            "format": file_data.get("format"),
            "error": file_data.get("error"),
            "content": file_content
        })
 
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"File not found: {file_id}")
            return json.dumps({
                "error": f"File not found: {file_id}",
                "status_code": 404
            })
        else:
            logger.error(f"HTTP error fetching file: {str(e)}")
            return json.dumps({
                "error": f"HTTP error: {str(e)}",
                "status_code": e.response.status_code
            })
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching file: {str(e)}")
        return json.dumps({
            "error": f"Request error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Unexpected error fetching file: {str(e)}")
        return json.dumps({
            "error": f"Unexpected error: {str(e)}"
        })
       
@mcp.tool()
def list_finetuning_files(ctx: Context, purpose: str = "fine-tune") -> str:
    """
    Lists all files available for fine-tuning in Azure OpenAI.
   
    Parameters:
    - purpose: Filter files by purpose (default: "fine-tune")
   
    Returns:
    - JSON string with list of files and their details
    """
    if not azure_endpoint or not api_key:
        return json.dumps({
            "error": "Missing required environment variables: 'AZURE_OPENAI_ENDPOINT' or 'AZURE_OPENAI_API_KEY'."
        })
   
    url = f"{azure_endpoint}/openai/files?api-version={api_version}"
    if purpose:
        url += f"&purpose={purpose}"
   
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
   
    try:
        logger.info(f"Listing files with purpose: {purpose}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
       
        files_data = response.json()
        files_list = []
       
        for file in files_data.get("data", []):
            files_list.append({
                "file_id": file.get("id"),
                "filename": file.get("filename"),
                "purpose": file.get("purpose"),
                "bytes": file.get("bytes"),
                "created_at": file.get("created_at"),
                "status": file.get("status")
            })
       
        return json.dumps({
            "files": files_list,
            "count": len(files_list),
            "purpose_filter": purpose
        })
       
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error listing files: {str(e)}")
        return json.dumps({
            "error": f"Request error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Unexpected error listing files: {str(e)}")
        return json.dumps({
            "error": f"Unexpected error: {str(e)}"
        })
 
 
@mcp.tool()
def create_finetuning_job(
    ctx: Context,
    job_type: str,
    training_file_id: str,
    validation_file_id: str = None,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict = None
) -> str:
    """
    Creates a fine-tuning job only if the file format is valid if supervised it should be supervised and not dpo and vice versa.
    NOTE: Before running this tool, the LLM must call validate_finetuning_file_whether_dpo_or_supervised whenever the user requests to create a finetuning job.
 
    Parameters:
    - job_type: Type of finetuning job ('supervised' or 'dpo')
    - training_file_id: ID of the training file
    - validation_file_id: Optional ID of the validation file
    - model: Base model to fine-tune (default: gpt-4o-mini-2024-07-18)
    - hyperparameters: Optional hyperparameters
 
    Returns:
    - JSON string with job result.
    """
 
    if not azure_endpoint or not api_key:
        return json.dumps({
            "error": "Missing required environment variables: 'AZURE_OPENAI_ENDPOINT' or 'AZURE_OPENAI_API_KEY'."
        })
   
    if job_type not in ['supervised', 'dpo']:
        return json.dumps({
            "error": "Invalid job_type. Must be 'supervised' or 'dpo'."
        })
   
    logger.info(f"Creating {job_type} finetuning job with file: {training_file_id}")
   
    url = f"{azure_endpoint}/openai/fine_tuning/jobs?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
   
    body = {
        "model": model,
        "training_file": training_file_id,
    }
   
    if validation_file_id:
        body["validation_file"] = validation_file_id
   
    if hyperparameters:
        body["hyperparameters"] = hyperparameters
    else:
        if job_type == 'dpo':
            body["hyperparameters"] = {
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 1.8
            }
        else:
            body["hyperparameters"] = {
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 2
            }
   
    if job_type == 'dpo':
        method = {"type": "dpo"}
        if hyperparameters and "dpo" in hyperparameters:
            method.update(hyperparameters["dpo"])
        body["method"] = method
   
    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
       
        if response.status_code >= 400:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
            logger.error(f"API error creating finetuning job: {error_message}")
           
            # Azure API will return specific format errors
            return json.dumps({
                "success": False,
                "error": error_message,
                "status_code": response.status_code,
                "error_details": error_data,
                "hint": f"See resource finetuning://format/{job_type} for expected format"
            })
       
        job_data = response.json()
        return json.dumps({
            "success": True,
            "job_id": job_data.get("id"),
            "status": job_data.get("status"),
            "job_type": job_type,
            "model": job_data.get("model"),
            "training_file": job_data.get("training_file"),
            "created_at": job_data.get("created_at")
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

@mcp.tool()
def execute_dynamic_swagger_action(ctx: Context, tool_name: str, **params) -> str:
    """
    Execute a dynamically generated tool from the Swagger specification.
    
    Args:
        tool_name: Name of the tool (operation ID)
        **params: Parameters for the API call
        
    Returns:
        JSON string with the API response
    """
    swagger_generator = get_swagger_generator()
    
    if swagger_generator is None:
        return json.dumps({
            "error": "No Swagger tools have been registered",
            "hint": "Ensure SWAGGER_PATH is set in your .env file"
        })
    
    # Check if params were wrapped in a 'params' key (common MCP pattern)
    if 'params' in params and isinstance(params['params'], dict):
        # Unwrap the parameters
        actual_params = params['params']
    else:
        # Use params as-is
        actual_params = params
    
    # Log the request details before calling the API
    try:
        if tool_name in swagger_generator.registered_tools:
            tool_info = swagger_generator.registered_tools[tool_name]["info"]
            base_url = swagger_generator._extract_base_url()
            path = tool_info.path
            method = tool_info.method
            
            logger.info(f"Tool: {tool_name}")
            logger.info(f"Base URL: {base_url}")
            logger.info(f"Path template: {path}")
            logger.info(f"Method: {method}")
            logger.info(f"Parameters provided: {actual_params}")
            
            # Check if all required path parameters are provided
            path_params_in_template = re.findall(r'\{(\w+)\}', path)
            logger.info(f"Path parameters expected: {path_params_in_template}")
            
            for param in path_params_in_template:
                if param not in actual_params:
                    logger.warning(f"Missing required path parameter: {param}")
        else:
            logger.error(f"Tool not found: {tool_name}")
    except Exception as e:
        logger.warning(f"Could not log Swagger API details: {str(e)}")

    # Use the unwrapped parameters
    return swagger_generator.execute_tool(tool_name, **actual_params)


@mcp.tool()
def list_dynamic_swagger_tools(ctx: Context) -> str:
    """
    List all dynamically registered tools from the Swagger specification.
    NOTE:
    # LLM Context Note:
    # When a user requests to "list all finetuning jobs", you (the LLM) must:
        # 1. Call `list_dynamic_swagger_tools(ctx)` to get all available Swagger-based tools.
        # 2. Parse the returned list and automatically select the tool whose name, path, or description matches the intent (such as "list finetuning jobs").
        # 3. Call `execute_dynamic_swagger_action(ctx, tool_name=...)` with the selected tool's operation ID.
        # This process is fully automated and does not require user or developer intervention.
        # If no matching Swagger tool is found, return an error message.
    
    # When a user requests to "get finetuning job events", you (the LLM) must:
        # 1. Call `list_dynamic_swagger_tools(ctx)` to get all available Swagger-based tools.
        # 2. Parse the returned list and automatically select the tool whose name, path, or description matches the intent (such as "get finetuning job events").
        # 3. Call `execute_dynamic_swagger_action(ctx, tool_name=..., job_id=...)` with the selected tool's operation ID and required parameters.
        # This process is fully automated and does not require user or developer intervention.
        # If no matching Swagger tool is found, return an error message.
    
    # When a user requests to "fetch finetuning job status", you (the LLM) must:
        # 1. Call `list_dynamic_swagger_tools(ctx)` to get all available Swagger-based tools.
        # 2. Parse the returned list and automatically select the tool whose name, path, or description matches the intent (such as "fetch finetuning job status").
        # 3. Call `execute_dynamic_swagger_action(ctx, tool_name=..., job_id=...)` with the selected tool's operation ID and required parameters.
        # This process is fully automated and does not require user or developer intervention.
        # If no matching Swagger tool is found, return an error message.
    Returns:
        JSON string with list of available tools
    """
    swagger_generator = get_swagger_generator()
    
    if swagger_generator is None:
        return json.dumps({
            "error": "No Swagger tools have been registered",
            "hint": "Ensure SWAGGER_PATH is set in your .env file"
        })
    
    tools_list = []
    for tool_name, tool_data in swagger_generator.registered_tools.items():
        info = tool_data["info"]
        tools_list.append({
            "name": tool_name,
            "method": info.method,
            "path": info.path,
            "description": info.description,
            "parameters": list(info.parameters["properties"].keys())
        })
    
    return json.dumps({
        "total_tools": len(tools_list),
        "tools": tools_list,
        "base_url": swagger_generator.base_url
    }, indent=2)