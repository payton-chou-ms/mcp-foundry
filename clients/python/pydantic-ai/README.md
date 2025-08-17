# MCP Client using PydanticAI

This is an MCP client written with PydanticAI that is intended to be used for demoing the Foundry MCP Server.

It comes with a helper MCP server that has tools for fetching remote URL contents that can be used in the service.

For the demo, we are going to use the sample dataset to interact with the Azure AI Search tools from the MCP service.

You can run the PydanticAI Sample code as follows:

```bash
git clone git@github.com:azure-ai-foundry/mcp-foundry.git

cd clients/python/pydantic-ai 

uv run main.py
```

Once you run the command, you can use prompts to interact with the MCP server.
