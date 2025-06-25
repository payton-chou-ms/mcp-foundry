import os
import json
import requests
from mcp_foundry.mcp_server import mcp
from dotenv import load_dotenv


load_dotenv()

@mcp.tool(description ="Fetch the status of a fine-tuning job using Azure OpenAI API")
def fetch_finetuning_status(job_id: str) -> str:
    """
    Fetches the status of a fine-tuning job using Azure OpenAI API.

    Parameters:
    - job_id: The ID of the fine-tuning job.

    Returns:
    - Job status information as a JSON string.
    """
    
    project_endpoint = os.environ.get("PROJECT_ENDPOINT", "").rstrip("/")
    api_version = os.environ.get("api_version")
    api_key = os.environ.get("FOUNDRY_API_KEY")

    if not project_endpoint or not api_key or not api_version:
        return json.dumps({
            "error": "Missing required environment variables: 'PROJECT_ENDPOINT', 'FOUNDRY_API_KEY', or 'api_version'."
        })

    url = f"{project_endpoint}/openai/fine_tuning/jobs/{job_id}?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        job_data = response.json()

        return json.dumps({
            "job_id": job_data.get("id"),
            "status": job_data.get("status"),
            "model": job_data.get("model"),
            "created_at": job_data.get("created_at"),
            "finished_at": job_data.get("finished_at"),
            "hyperparameters": job_data.get("hyperparameters")
        })
    except Exception as e:
        return json.dumps({"error": f"Error fetching fine-tuning job status: {str(e)}"})
