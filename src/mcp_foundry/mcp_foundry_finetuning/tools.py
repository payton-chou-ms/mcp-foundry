import os
import json
import logging
import sys
import requests
from mcp_foundry.mcp_server import mcp
from mcp.server.fastmcp import Context
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from urllib.parse import urljoin
import re
import yaml
 
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
def generate_swagger_Actions(ctx: Context, swagger_path: str = "swagger.yaml") -> str:
    """
    Generates all actionable endpoints from a provided swagger file.
    Supports advanced use cases by exposing API actions dynamically.

    Parameters:
    - swagger_path: Path to the swagger file (default: "swagger.yaml")

    Returns:
    - JSON string listing all endpoints with method, path, operationId, summary, and parameters.
    """
    try:
        with open(swagger_path, "r", encoding="utf-8") as f:
            swagger = yaml.safe_load(f)
    except Exception as e:
        return json.dumps({"error": f"Failed to load swagger file: {str(e)}"})

    endpoints = []
    paths = swagger.get("paths", {})
    components = swagger.get("components", {})
    param_defs = components.get("parameters", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            if method not in ["get", "post", "put", "delete", "patch"]:
                continue
            endpoint = {
                "method": method.upper(),
                "path": path,
                "operationId": details.get("operationId"),
                "summary": details.get("summary"),
                "description": details.get("description"),
                "parameters": []
            }
            for param in details.get("parameters", []):
                if isinstance(param, dict) and "$ref" in param:
                    ref = param["$ref"]
                    ref_name = ref.split("/")[-1]
                    param_detail = param_defs.get(ref_name, {})
                    endpoint["parameters"].append({
                        "name": param_detail.get("name"),
                        "in": param_detail.get("in"),
                        "required": param_detail.get("required"),
                        "type": param_detail.get("schema", {}).get("type"),
                        "description": param_detail.get("description"),
                        "example": param_detail.get("example")
                    })
                elif isinstance(param, dict):
                    endpoint["parameters"].append({
                        "name": param.get("name"),
                        "in": param.get("in"),
                        "required": param.get("required"),
                        "type": param.get("schema", {}).get("type"),
                        "description": param.get("description"),
                        "example": param.get("example")
                    })
            endpoints.append(endpoint)
    return json.dumps({"endpoints": endpoints}, indent=2)

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
    When starting a fine-tuning job, the LLM will automatically check the file structure and decide the job type:
    - If both 'preferred_output' and 'non_preferred_output' are present, DPO will be used. If the user requests supervised but the file is DPO, give him 2 choices whether to change the file to supervised format or change job type to DPO.
    - If it doesn't contain 'preferred_output' or 'non_preferred_output', or both supervised will be used. If the user requests DPO but the file is supervised, give him 2 choices whether to change the file to DPO format or change job type to supervised.
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
    NOTE: Before running this tool, the LLM must call validate_finetuning_file_whether_dpo_or_supervised
 
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
    
# Add these imports at the top of the file (after existing imports)
# Add this class and functions after your existing tool definitions

class SwaggerToolGenerator:
    """
    Dynamically generates MCP tools from Swagger/OpenAPI specifications.
    Specifically optimized for Azure OpenAI APIs.
    """
   
    def __init__(self, swagger_file_path: str = None, swagger_url: str = None, swagger_content: str = None):
        self.swagger_data = self._load_swagger(swagger_file_path, swagger_url, swagger_content)
        self.base_url = self._extract_base_url()
        self.api_key = api_key  # Use the global api_key from your existing code
        self.api_version = api_version  # Use the global api_version
        self.registered_tools = {}
       
    def _load_swagger(self, file_path: str = None, url: str = None, content: str = None) -> Dict:
        """Load Swagger specification from various sources."""
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                try:
                    return yaml.safe_load(content)
                except Exception as e:
                    raise ValueError(f"Failed to parse swagger content: {str(e)}")
                   
        elif file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Swagger file not found: {file_path}")
           
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        return yaml.safe_load(content)
                    except Exception as e:
                        raise ValueError(f"Failed to parse swagger file: {str(e)}")
                       
        elif url:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
               
                content_type = response.headers.get('content-type', '').lower()
                if 'json' in content_type:
                    return response.json()
                else:
                    try:
                        return response.json()
                    except:
                        try:
                            return yaml.safe_load(response.text)
                        except Exception as e:
                            raise ValueError(f"Failed to parse swagger from URL: {str(e)}")
                           
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Failed to fetch swagger from URL: {str(e)}")
        else:
            raise ValueError("Must provide either swagger_file_path, swagger_url, or swagger_content")
   
    def _extract_base_url(self) -> str:
        """Extract base URL from Swagger specification."""
        # OpenAPI 3.0
        if "servers" in self.swagger_data and self.swagger_data["servers"]:
            server = self.swagger_data["servers"][0]
            url = server.get("url", "")
           
            # Handle variable substitution for Azure endpoints
            if "variables" in server:
                for var_name, var_info in server["variables"].items():
                    placeholder = f"{{{var_name}}}"
                    if placeholder in url:
                        # Use environment variable or default
                        replacement = azure_endpoint or var_info.get("default", "")
                        url = url.replace(placeholder, replacement)
           
            return url
           
        # Swagger 2.0
        elif "host" in self.swagger_data:
            scheme = self.swagger_data.get("schemes", ["https"])[0]
            host = self.swagger_data.get("host", "")
            base_path = self.swagger_data.get("basePath", "")
            return f"{scheme}://{host}{base_path}"
           
        return azure_endpoint or "https://your-resource.openai.azure.com"
   
    def _resolve_reference(self, ref: str) -> Dict:
        """Resolve $ref references in the Swagger document."""
        if not ref.startswith("#/"):
            return {}
           
        path_parts = ref[2:].split("/")
        result = self.swagger_data
       
        for part in path_parts:
            if isinstance(result, dict) and part in result:
                result = result[part]
            else:
                return {}
               
        return result
   
    def _build_parameter_schema(self, parameters: List[Dict]) -> Dict[str, Any]:
        """Build parameter schema for tool registration."""
        properties = {}
        required = []
       
        for param in parameters:
            # Resolve reference if needed
            if "$ref" in param:
                param = self._resolve_reference(param["$ref"])
           
            param_name = param.get("name", "")
            param_in = param.get("in", "")
           
            # Skip api-version as it's handled automatically for Azure
            if param_name == "api-version":
                continue
           
            # Build parameter schema
            param_schema = {
                "type": param.get("schema", {}).get("type", "string"),
                "description": param.get("description", "")
            }
           
            # Add enum values if present
            if "enum" in param.get("schema", {}):
                param_schema["enum"] = param["schema"]["enum"]
           
            # Add pattern if present
            if "pattern" in param.get("schema", {}):
                param_schema["pattern"] = param["schema"]["pattern"]
           
            # Add example if present
            if "example" in param:
                param_schema["example"] = param["example"]
            elif "example" in param.get("schema", {}):
                param_schema["example"] = param["schema"]["example"]
           
            properties[param_name] = param_schema
           
            if param.get("required", False):
                required.append(param_name)
       
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
   
    def _create_tool_function(self, path: str, method: str, operation: Dict) -> Callable:
        """Create a tool function for a specific API operation."""
       
        def tool_function(ctx, **kwargs) -> str:
            """Dynamically generated tool function."""
            try:
                # Build URL
                url = urljoin(self.base_url, path)
               
                # Process path parameters
                path_params = {}
                query_params = {"api-version": self.api_version}
                headers = {
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                }
               
                # Extract parameters
                for param in operation.get("parameters", []):
                    if "$ref" in param:
                        param = self._resolve_reference(param["$ref"])
                   
                    param_name = param.get("name", "")
                    param_in = param.get("in", "")
                   
                    if param_name in kwargs:
                        if param_in == "path":
                            path_params[param_name] = kwargs[param_name]
                        elif param_in == "query":
                            query_params[param_name] = kwargs[param_name]
                        elif param_in == "header":
                            headers[param_name] = kwargs[param_name]
               
                # Replace path parameters
                for param_name, param_value in path_params.items():
                    url = url.replace(f"{{{param_name}}}", str(param_value))
               
                # Handle request body
                json_data = None
                if "requestBody" in operation and method.lower() in ["post", "put", "patch"]:
                    # For simplicity, assume JSON content
                    body_params = {k: v for k, v in kwargs.items()
                                 if k not in path_params and k not in query_params}
                    if body_params:
                        json_data = body_params
               
                # Make the request
                logger.info(f"Executing {method.upper()} {url}")
               
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    params=query_params,
                    json=json_data,
                    timeout=30
                )
               
                # Handle response
                if response.status_code >= 400:
                    error_detail = {
                        "error": f"HTTP {response.status_code}",
                        "message": response.text,
                        "url": url,
                        "method": method.upper()
                    }
                    return json.dumps(error_detail)
               
                # Return appropriate response format
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    return json.dumps(response.json(), indent=2)
                elif "text/csv" in content_type:
                    return json.dumps({
                        "content_type": "text/csv",
                        "content": response.text
                    })
                else:
                    return json.dumps({
                        "content_type": content_type,
                        "content": response.text
                    })
                   
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                return json.dumps({
                    "error": f"Request error: {str(e)}",
                    "url": url if 'url' in locals() else path,
                    "method": method.upper()
                })
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return json.dumps({
                    "error": f"Unexpected error: {str(e)}",
                    "method": method.upper()
                })
       
        return tool_function
   
    def generate_and_register_tools(self) -> Dict[str, Any]:
        """Generate and register all tools from the Swagger specification."""
       
        results = {
            "api_info": {
                "title": self.swagger_data.get("info", {}).get("title", "Unknown API"),
                "version": self.swagger_data.get("info", {}).get("version", "1.0.0"),
                "base_url": self.base_url
            },
            "registered_tools": [],
            "errors": []
        }
       
        paths = self.swagger_data.get("paths", {})
       
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
           
            for method in ["get", "post", "put", "patch", "delete"]:
                if method not in path_item:
                    continue
               
                operation = path_item[method]
                operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
               
                # Clean operation ID for use as function name
                clean_operation_id = re.sub(r'[^a-zA-Z0-9_]', '_', operation_id)
               
                try:
                    # Build parameter schema
                    parameters = operation.get("parameters", [])
                    param_schema = self._build_parameter_schema(parameters)
                   
                    # Create tool function
                    tool_func = self._create_tool_function(path, method, operation)
                   
                    # Generate tool description
                    description = f"{operation.get('summary', '')}. {operation.get('description', '')}".strip()
                    if not description:
                        description = f"Execute {method.upper()} {path}"
                   
                    # Store tool info
                    tool_info = {
                        "name": clean_operation_id,
                        "operation_id": operation_id,
                        "method": method.upper(),
                        "path": path,
                        "description": description,
                        "parameters": param_schema,
                        "tags": operation.get("tags", [])
                    }
                   
                    self.registered_tools[clean_operation_id] = {
                        "function": tool_func,
                        "info": tool_info
                    }
                   
                    results["registered_tools"].append(tool_info)
                   
                    logger.info(f"Registered tool: {clean_operation_id}")
                   
                except Exception as e:
                    error_msg = f"Failed to register {operation_id}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
       
        return results
   
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a registered tool by name."""
        if tool_name not in self.registered_tools:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})
       
        tool_func = self.registered_tools[tool_name]["function"]
        return tool_func(None, **kwargs)  # ctx=None for standalone execution


# Global variable to store the swagger generator instance
_swagger_generator = None


def register_swagger_tools(swagger_file_path: str = None, swagger_url: str = None, swagger_content: str = None) -> Dict[str, Any]:
    """
    Main function to register all tools from a Swagger specification.
   
    Returns:
        Dictionary with registration results and tool information
    """
    try:
        generator = SwaggerToolGenerator(swagger_file_path, swagger_url, swagger_content)
        results = generator.generate_and_register_tools()
       
        # Store generator instance for later use
        global _swagger_generator
        _swagger_generator = generator
       
        return results
       
    except Exception as e:
        logger.error(f"Failed to register swagger tools: {str(e)}")
        return {
            "error": str(e),
            "registered_tools": [],
            "errors": [str(e)]
        }


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
    global _swagger_generator
   
    if _swagger_generator is None:
        return json.dumps({"error": "No Swagger tools have been registered yet. Please call register_swagger_tools first."})
   
    return _swagger_generator.execute_tool(tool_name, **params)


@mcp.tool()
def list_dynamic_swagger_tools(ctx: Context) -> str:
    """
    List all dynamically registered tools from the Swagger specification.
    
    Returns:
        JSON string with list of available tools and their details
    """
    global _swagger_generator
    
    if _swagger_generator is None:
        return json.dumps({
            "error": "No Swagger tools have been registered yet",
            "hint": "Call register_swagger_tools with a swagger file path first"
        })
    
    tools_list = []
    for tool_name, tool_data in _swagger_generator.registered_tools.items():
        info = tool_data["info"]
        tools_list.append({
            "name": tool_name,
            "method": info["method"],
            "path": info["path"],
            "description": info["description"],
            "parameters": list(info["parameters"]["properties"].keys()),
            "required_params": info["parameters"]["required"]
        })
    
    return json.dumps({
        "total_tools": len(tools_list),
        "tools": tools_list,
        "base_url": _swagger_generator.base_url,
        "api_info": {
            "title": _swagger_generator.swagger_data.get("info", {}).get("title", "Unknown API"),
            "version": _swagger_generator.swagger_data.get("info", {}).get("version", "1.0.0")
        }
    }, indent=2)


@mcp.tool()
def register_swagger_from_file(ctx: Context, swagger_path: str) -> str:
    """
    Register all tools from a Swagger/OpenAPI specification file.
    
    Args:
        swagger_path: Path to the swagger file (JSON or YAML)
        
    Returns:
        JSON string with registration results
    """
    try:
        results = register_swagger_tools(swagger_file_path=swagger_path)
        
        if "error" in results:
            return json.dumps(results)
            
        return json.dumps({
            "success": True,
            "message": f"Successfully registered {len(results['registered_tools'])} tools",
            "api_info": results["api_info"],
            "tools_count": len(results["registered_tools"]),
            "errors_count": len(results["errors"]),
            "hint": "Use list_dynamic_swagger_tools to see all available tools"
        })
        
    except Exception as e:
        logger.error(f"Failed to register swagger tools: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })


# Auto-register tools on module import if swagger file is available
swagger_path = os.environ.get("SWAGGER_PATH", "")
if swagger_path and os.path.exists(swagger_path):
    logger.info(f"Auto-registering tools from {swagger_path}")
    try:
        register_swagger_tools(swagger_file_path=swagger_path)
        logger.info("Successfully auto-registered swagger tools")
    except Exception as e:
        logger.error(f"Failed to auto-register swagger tools: {str(e)}")