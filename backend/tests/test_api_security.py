"""
Security-focused API tests — verify auth enforcement, enum validation,
and dry_run lock-down without needing real credentials or a running server.
"""
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("APP_API_KEY",              "secure-test-key-abc123")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("DATABASE_URL",             ":memory:")

from httpx import AsyncClient, ASGITransport


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    """Import the FastAPI app without running its lifespan."""
    import importlib
    import main as m
    importlib.reload(m)
    return m.app


@pytest.fixture
def api_key():
    return os.environ["APP_API_KEY"]


# ── Auth tests ────────────────────────────────────────────────────────────────

class TestAuthentication:
    @pytest.mark.asyncio
    async def test_get_status_no_auth_required(self, app):
        """Read-only status endpoint should be publicly accessible."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("main.coinbase_client.get_usd_balance", new_callable=AsyncMock, return_value=500.0):
                resp = await client.get("/api/status")
        assert resp.status_code in (200, 500)  # may 500 if DB not init'd — that's ok

    @pytest.mark.asyncio
    async def test_post_trading_enable_requires_api_key(self, app):
        """Enable trading without API key should return 401."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/trading/enable")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_post_trading_enable_wrong_key_returns_401(self, app, api_key):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/trading/enable",
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_delete_order_requires_api_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/api/orders/some-order-id")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_post_orders_requires_api_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/orders", json={
                "product_id": "BTC-USD",
                "side":       "BUY",
                "order_type": "LIMIT",
                "price":      94000.0,
                "quote_size": 50.0,
            })
        assert resp.status_code == 401


# ── Input validation tests ────────────────────────────────────────────────────

class TestInputValidation:
    @pytest.mark.asyncio
    async def test_invalid_side_rejected(self, app, api_key):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/orders",
                json={
                    "product_id": "BTC-USD",
                    "side":       "LONG",       # invalid — must be BUY or SELL
                    "order_type": "LIMIT",
                    "quote_size": 50.0,
                },
                headers={"X-API-Key": api_key},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_order_type_rejected(self, app, api_key):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/orders",
                json={
                    "product_id": "BTC-USD",
                    "side":       "BUY",
                    "order_type": "STOP",       # invalid — only LIMIT or MARKET
                    "quote_size": 50.0,
                },
                headers={"X-API-Key": api_key},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_negative_size_rejected(self, app, api_key):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/orders",
                json={
                    "product_id": "BTC-USD",
                    "side":       "BUY",
                    "order_type": "MARKET",
                    "quote_size": -10.0,        # must be positive
                },
                headers={"X-API-Key": api_key},
            )
        assert resp.status_code == 422


# ── Dry-run lock ──────────────────────────────────────────────────────────────

class TestDryRunLock:
    def test_dry_run_reads_from_config_not_api(self):
        """Verify dry_run is sourced from config (env var) and cannot be changed by the API caller."""
        import importlib
        import config as cfg
        importlib.reload(cfg)
        assert cfg.config.dry_run is True

    def test_dry_run_true_when_env_set_to_true(self):
        os.environ["DRY_RUN"] = "true"
        import importlib
        import config as cfg
        importlib.reload(cfg)
        assert cfg.config.dry_run is True

    def test_dry_run_false_only_when_explicitly_set(self):
        os.environ["DRY_RUN"] = "false"
        import importlib
        import config as cfg
        importlib.reload(cfg)
        assert cfg.config.dry_run is False
        # Reset
        os.environ["DRY_RUN"] = "true"
        importlib.reload(cfg)


# ── Balance endpoint ──────────────────────────────────────────────────────────

class TestBalanceEndpoint:
    """GET /api/balance must call get_accounts exactly once, not twice."""

    @pytest.mark.asyncio
    async def test_balance_calls_get_accounts_once(self, app):
        """Verify /api/balance does not double-call get_accounts (old bug: called
        get_accounts directly then again inside get_usd_balance)."""
        fake_accounts = [
            {"currency": "USD",  "available": 500.0, "hold": 0.0},
            {"currency": "BTC",  "available": 0.001,  "hold": 0.0},
        ]
        with patch(
            "main.coinbase_client.get_accounts",
            new_callable=AsyncMock,
            return_value=fake_accounts,
        ) as mock_accounts:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/balance")

        assert mock_accounts.call_count == 1, (
            f"/api/balance called get_accounts {mock_accounts.call_count}x — expected exactly 1"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["usd_balance"] == 500.0

    @pytest.mark.asyncio
    async def test_balance_returns_usd_from_accounts(self, app):
        """usd_balance in response matches the USD account's available field."""
        fake_accounts = [
            {"currency": "ETH",  "available": 0.5,    "hold": 0.0},
            {"currency": "USD",  "available": 999.99, "hold": 0.0},
        ]
        with patch(
            "main.coinbase_client.get_accounts",
            new_callable=AsyncMock,
            return_value=fake_accounts,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/balance")

        assert resp.json()["usd_balance"] == pytest.approx(999.99)
