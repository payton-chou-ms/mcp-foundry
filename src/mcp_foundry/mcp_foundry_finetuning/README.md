# Extending the Finetuning Tools in MCP Foundry

This guide explains how to add new tools to the `mcp_foundry_finetuning` folder, including both static MCP tools and dynamic tools generated from a Swagger/OpenAPI specification.

## 1. Adding Static MCP Tools

Static tools are Python functions decorated with `@mcp.tool()` in `tools.py`. These are registered with the MCP server and can be called directly.

### Steps to Add a Static Tool

1. **Open** `src/mcp_foundry/mcp_foundry_finetuning/tools.py`.
2. **Define your function** and decorate it with `@mcp.tool()`. The function should accept a `Context` object as the first argument.
3. **Implement your logic**. Use environment variables for configuration (see existing tools for examples).
4. **Return results** as a JSON-serializable object or string.

#### Example
```python
from mcp_foundry.mcp_server import mcp
from mcp.server.fastmcp import Context

@mcp.tool()
def my_new_tool(ctx: Context, param1: str) -> str:
    # Your logic here
    return f"Received: {param1}"
```

5. **Test your tool** by running the MCP server and calling the tool via the client.

## 2. Adding Dynamic Tools from Swagger/OpenAPI

Dynamic tools are generated automatically from the Swagger specification (see `swagger.yaml`). The generator reads the spec and registers tools for each operation.

### Steps to Add a Dynamic Tool

1. **Edit** `swagger.yaml` in the project root.
2. **Add a new path/operation** following the OpenAPI 3.0 format. Each operation should have a unique `operationId`.
3. **(Optional) Add schemas** for request/response bodies in the `components` section.
4. **Ensure** the `SWAGGER_PATH` environment variable is set to the path of your `swagger.yaml` file (usually in your `.env`).
5. **Restart** the MCP server. The dynamic tool will be auto-registered if the Swagger spec is valid.

#### Example (swagger.yaml)
```yaml
paths:
  /openai/fine_tuning/jobs/my_custom:
    get:
      operationId: myCustomTool
      summary: My custom tool
      description: Does something custom
      responses:
        '200':
          description: Success
```

### How Dynamic Tools Work
- The generator in `swagger.py` parses the YAML and registers a tool for each operation.
- You can list all dynamic tools using the `list_dynamic_swagger_tools` MCP tool.
- Call a dynamic tool by its `operationId` using the `execute_dynamic_swagger_action` MCP tool.

## 3. Tips and Best Practices
- **Use clear, unique operationIds** in Swagger for dynamic tools.
- **Document parameters and responses** in both Python and YAML.
- **Check logs** for errors if your tool does not appear or fails to register.
- **Keep your Swagger spec valid** (use an online validator if needed).

## 4. Useful References
- See `tools.py` for static tool examples.
- See `swagger.yaml` for dynamic tool definitions.
- See `swagger.py` for the dynamic tool generator logic.

---
