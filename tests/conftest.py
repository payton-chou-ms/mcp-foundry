import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration")

def pytest_collection_modifyitems(config, items):
    run_integration = config.getoption("--runintegration")

    if run_integration:
        return  # allow all tests

    skip_marker = pytest.mark.skip(reason="Skipped integration test (use --runintegration to enable)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)

def pytest_addoption(parser):
    parser.addoption(
        "--runintegration", action="store_true", default=False, help="Run integration tests"
    )
