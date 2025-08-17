import pytest
from mcp_foundry.mcp_foundry_model.models import ModelDetails, DeploymentOption
from mcp_foundry.mcp_foundry_model.tools import get_model_details_and_code_samples

def _mock_ctx():
    """Mock context for testing."""
    class MockContext:
        def __init__(self):
            self.session = MockSession()
    class MockSession:
        def __init__(self):
            self._client_params = MockClientParams()
    class MockClientParams:
        def __init__(self):
            self.clientInfo = MockClientInfo()
    class MockClientInfo:
        def __init__(self):
            self.name = "TestClient"
            self.version = "1.0.0"
    return MockContext()

@pytest.mark.asyncio
async def test_get_model_details_from_model_catalog_gh_model():
    mock_ctx = _mock_ctx()
    models = await get_model_details_and_code_samples('o3',mock_ctx)
    assert isinstance(models, ModelDetails)
    assert models.type == DeploymentOption.FREE_PLAYGROUND

    
@pytest.mark.asyncio
async def test_get_model_details_from_model_catalog_gh_model_2():
    mock_ctx = _mock_ctx()
    models = await get_model_details_and_code_samples('Phi-4-reasoning',mock_ctx)
    assert isinstance(models, ModelDetails)
    assert models.type == DeploymentOption.FREE_PLAYGROUND

@pytest.mark.asyncio
async def test_get_model_details_from_model_catalog_labs_model():
    mock_ctx = _mock_ctx()
    models = await get_model_details_and_code_samples('omniparserv2',mock_ctx)
    assert isinstance(models, ModelDetails)
    assert models.type == DeploymentOption.LABS

@pytest.mark.asyncio
async def test_get_model_details_from_paid_openai_model():
    mock_ctx = _mock_ctx()
    models = await get_model_details_and_code_samples('gpt-image-1',mock_ctx)
    assert isinstance(models, ModelDetails)
    assert models.type == DeploymentOption.OPENAI

@pytest.mark.asyncio
async def test_get_model_details_from_standard_paygo_model():
    mock_ctx = _mock_ctx()
    models = await get_model_details_and_code_samples('BioEmu',mock_ctx)
    assert isinstance(models, ModelDetails)
    assert models.type == DeploymentOption.SERVERLESS_ENDPOINT

@pytest.mark.asyncio
async def test_get_model_details_from_model_catalog_unsupported_model():
    mock_ctx = _mock_ctx()
    models = await get_model_details_and_code_samples('unsupported_model',mock_ctx)
    assert models == "Model 'unsupported_model' not found in the catalog."