from mcp_foundry.mcp_server import mcp

@mcp.resource("examples://python-mcp-client-pydantic-ai",
                  description="A resource showing how to communicate with the MCP server using Pydantic AI",
                  mime_type="text/markdown")
async def sample_python_mcp_client_resource() -> str:
    return """
    ## MCP Client Written with Pydantic AI
    
    This is an MCP client written with Pydantic AI that is intended to be used for checking out all the capabilities of the MCP service

    https://github.com/azure-ai-foundry/mcp-foundry/sample-python-clients
    """