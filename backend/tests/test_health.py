import pytest
from httpx import ASGITransport, AsyncClient

from good_driver.app import create_app


@pytest.fixture
def app():
    return create_app()


async def test_health(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
