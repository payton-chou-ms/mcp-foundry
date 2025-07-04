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

