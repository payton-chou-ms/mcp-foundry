# Client examples

## Prepare the environment to run the server

Recommended way to run this MCP server is to use `uv` / `uvx`.

To install `uv` / `uvx`, refer to [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

For example, in Linux/macOS,

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or, in Windows,

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you have Python installed, you can also install `uv` / `uvx` using `pipx` or `pip`:

```bash
# pipx recommended to install uv into an isolated environment
pipx install uv
# or `pip install uv`
```

> [!NOTE]
> - `uvx` is a simple alias to `uv tool run` created for convenience. By installing `uv`, you will also have `uvx` available.

## Run the server to use with Visual Studio Code

For Visual Studio Code, simply use the dedicated configuration file (`.vscode/mcp.json`) to run and/or connect to the MCP server. Visual Studio Code supports both Standard Input/Output (`stdio`) and Server-Sent Events (`sse`) modes.

### MCP configuration examples

#### Quick start example

The following `vscode/mcp.json` allows downloading and running the server from the remote URL.

```json
{
    "servers": {
        "mcp_foundry_server": {
            "type": "stdio",
            "command": "uvx",
            "args": [
                "--prerelease=allow",
                "--from",
                "git+https://github.com/azure-ai-foundry/mcp-foundry.git",
                "run-azure-ai-foundry-mcp",
                "--envFile",
                "${workspaceFolder}/.env"
            ]
        }
    }
}
```

> [!NOTE]
> - The server can take `.env` file as an argument to load environment variables from it. You can use the `--envFile` option to specify the path to the `.env` file. If the file is not found, the server will still start without loading it and will support all capabilities that do not require values passed via environment variables.

#### Other scenarios

- If you want to ensure it always download and run the latest version of the MCP server, you can use the `--no-cache` option.

    <details>
    <summary>Use with --no-cache option</summary>
    
    ```json
    {
        "servers": {
            "mcp_foundry_server": {
                "type": "stdio",
                "command": "uvx",
                "args": [
                    "--no-cache",
                    "--prerelease=allow",
                    "--from",
                    "git+https://github.com/azure-ai-foundry/mcp-foundry.git",
                    "run-azure-ai-foundry-mcp"
                ]
            }
        }
    }
    ```
    
    </details>

- You can run the server manually using the command line with SSE (server-sent events) mode, and configure `.vscode/mcp.json` to use the SSE transport.

    <details>
    <summary>Use with SSE transport</summary>
    
    First run the server using the command line:
    
    ```bash
    uvx --prerelease=allow --from git+https://github.com/azure-ai-foundry/mcp-foundry.git run-azure-ai-foundry-mcp --transport sse
    ```
    
    > [!NOTE]
    > - You can add `--no-cache` or `--envFile` option as you need.
    
    Then configure the `.vscode/mcp.json` to use the SSE transport:
    
    ```json
    {
        "servers": {
            "mcp_foundry_server": {
                "type": "sse",
                "url": "http://localhost:8000/sse"
            }
        }
    }
    ```
    
    </details>

- You can run the server from your local file system, instead of a remote URL.

    <details>
    <summary>Use with local clone</summary>
    
    First clone the repo to your local file system:
    
    ```bash
    git clone https://github.com/azure-ai-foundry/mcp-foundry.git
    ```
    
    Then use the following `.vscode/mcp.json` to run the server:
    
    ```json
    {
        "servers": {
            "mcp_foundry_server": {
                "type": "stdio",
                "command": "uvx",
                "args": [
                    "--prerelease=allow",
                    "--from",
                    "./path/to/local/repo",
                    "run-azure-ai-foundry-mcp"
                ]
            }
        }
    }
    ```
    
    </details>

> [!NOTE]
> - Role of `.vscode/mcp.json` is to configure the MCP server for Visual Studio Code. For `stdio` mode, it helps starting the server and connecting to the server. For `sse` mode, it helps connecting to the server that is already running.
> - Above examples are provided for your reference. You can modify the command and arguments as per your requirements.
> - To learn more about the transport modes supported by GitHub Copilot and its configuration format, refer to [Configuration format](https://code.visualstudio.com/docs/copilot/chat/mcp-servers#_configuration-format).

### Sample mcp.json files

For convenience, we provide a few samples of MCP configuration files for VS Code in the `mcp-configs` folder. You can use them as a reference to create your own configuration file `.vscode/mcp.json`.

- [mcp.stdio.uvx.local.json](./vscode/mcp-configs/mcp.stdio.uvx.local.json)
- [mcp.stdio.uvx.remote.json](./vscode/mcp-configs/mcp.stdio.uvx.remote.json)
- [mcp.sse.json](./vscode/mcp-configs/mcp.sse.json)

## Run the server in SSE mode for other MCP Clients

You can run the server manually using the command line with SSE (server-sent events) mode, either from remote URL or a cloned repo.

Below is an example command to run the server using the remote URL:

```bash
uvx --prerelease=allow --from git+https://github.com/azure-ai-foundry/mcp-foundry.git run-azure-ai-foundry-mcp --transport sse
```

> [!NOTE]
> - You can add `--no-cache` or `--envFile` option as you need, or even run it from a cloned repo. See [Other scenarios](#other-scenarios) for reference.

Once the server is up and running, you can configure the MCP client to use the SSE transport by specifying the URL of the server `http://localhost:8000/sse`.

### Sample MCP Client app using SSE transport

For your reference, we provide a [sample MCP client app based on PydanticAI](./python/pydantic-ai/README.md) that uses SSE transport.

## Troubleshooting

- Server fails to start because uvx, uv are not available
  - Refer to [Installing uv](https://docs.astral.sh/uv/getting-started/installation/) to install and fix for your environment.
