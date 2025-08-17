from mcp_foundry.mcp_server import mcp

@mcp.prompt(description="A prompt to list the names of all the indices")
async def list_all_indices_prompt() -> str:
    return "List all the indices by name"

@mcp.prompt(description="A prompt to retrieve the schema details of all the indices")
async def list_all_indices_details_prompt() -> str:
    return "Show the schema details of all the indexes"

@mcp.prompt(description="Get the detail for a specific schema")
async def retrieve_index_schema_prompt(index_name: str) -> str:
    return f"Show the for the {index_name} index"

@mcp.prompt(description="Display the contents of a local file")
async def fetch_local_file_contents_prompt(file_path: str) -> str:
    return f"Display the contents of the local file {file_path}"

@mcp.prompt(description="Display the contents of a URL")
async def fetch_url_contents_prompt(url: str) -> str:
    return f"Display the contents of the file {url}"

@mcp.prompt(description="Creates an index matching the schema of a JSON file (local file or URL)")
async def create_index_from_file_analysis_prompt(index_name: str, url: str) -> str:
    return f"Create an index called '{index_name}' that is compatible with the JSON file contents in the file {url}"

@mcp.prompt(description="Updates the index definition for a specific field")
async def modify_index_field_definition_prompt(index_name: str, field_name: str) -> str:
    return f"Modify the index '{index_name}' and make the {field_name} retrievable, searchable and filterable"

@mcp.prompt(description="Removes a specific index")
async def remove_index_definition_prompt(index_name: str) -> str:
    return f"Remove the '{index_name}' index"

@mcp.prompt(description="Adds the contents of a JSON file (local file or URL) to the specified index")
async def add_document_from_file_analysis_prompt(index_name: str, url: str) -> str:
    return f"Add a document or documents to the '{index_name}' index using the contents of the file {url}"

@mcp.prompt(description="Remove a document from the index")
async def remove_document_prompt(index_name: str, id: str) -> str:
    return f"""
    Remove a document from the '{index_name}' index matching id '{id}'
    Remove all documents from the '{index_name}' where the preferred language is French
    Remove all documents from the '{index_name}' where the sign up date is March 30th 2025
    """

@mcp.prompt(description="Queries the index")
async def search_index_prompt(index_name: str, id: str) -> str:
    return f"""

    - Show all documents from the '{index_name}' index
    - Show all documents from the '{index_name}' where the preferred language is French
    - Show all documents from the '{index_name}' where the sign up date is March 30th 2025
    """

@mcp.prompt(description="How many documents are in a specific document")
async def get_document_count_prompt(index_name: str, id: str) -> str:
    return f"How many documents are in the '{index_name}' index"

@mcp.prompt(description="List the names of the indexers in AI Search")
async def list_indexers_prompt() -> str:
    return f"List the names of the indexers in AI Search"

@mcp.prompt(description="Get details about a specific indexer")
async def get_indexer_detail_prompt(name: str) -> str:
    return f"Show the details for the '{name}' indexer"

@mcp.prompt(description="Creates and indexer with a datasource")
async def create_indexer_datasource_prompt(indexer_name: str, data_source_name: str) -> str:
    return f"Create an indexer named '{indexer_name}' with field mappings using the data source '{data_source_name}'"

@mcp.prompt(description="Creates and indexer with a datasource and skill set")
async def create_indexer_datasource_skill_set_prompt(indexer_name: str, data_source_name: str,
                                                     skill_set_name: str) -> str:
    return f"Create an indexer named '{indexer_name}' with field mappings using the data source '{data_source_name}' and skillset '{skill_set_name}'"

@mcp.prompt(description="List all the data sources and skill sets")
async def list_skills_and_data_sources_prompt() -> str:
    return "List all the skill sets and data sources"

@mcp.prompt(description="Show details for a specific data source")
async def get_data_source_details_prompt(name: str) -> str:
    return f"Show details for the '{name}' data source"

@mcp.prompt(description="Show details for a specific skill set")
async def get_skillset_details_prompt(name: str) -> str:
    return f"Show details for the '{name}' skillset"