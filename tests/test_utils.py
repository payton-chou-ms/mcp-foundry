import pytest
from mcp_foundry.mcp_foundry_model.models import ModelsList
from mcp_foundry.mcp_foundry_model.utils import get_models_list

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

def test_get_models_list_no_filters():
    mock_ctx = _mock_ctx()
    models = get_models_list(mock_ctx)
    assert isinstance(models, ModelsList)

def test_get_models_list_free_playground():
    mock_ctx = _mock_ctx()
    models = get_models_list(mock_ctx, search_for_free_playground=True)
    assert isinstance(models, ModelsList)