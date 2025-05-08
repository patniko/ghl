from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


# Basic Test
def test_root():
    response = client.get("/")
    assert response.status_code == 200

    # Check that response is JSON and contains expected fields
    json_response = response.json()
    assert "status" in json_response
    assert json_response["status"] == "healthy"
    assert "version" in json_response
    assert "name" in json_response
    assert "environment" in json_response
