import json
import time

import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from AgentCrew.modules.mcpclient.auth import InlineTokenStorage
from AgentCrew.modules.mcpclient.config import (
    MCPConfigManager,
    MCPOAuthOverrideConfig,
    MCPServerConfig,
)
from AgentCrew.modules.mcpclient.service import MCPService


class FakeTokenStorage:
    def __init__(self, tokens=None, client_info=None):
        self.tokens = tokens
        self.client_info = client_info
        self.set_tokens_calls = []
        self.set_client_info_calls = []

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens):
        self.tokens = tokens
        self.set_tokens_calls.append(tokens)

    async def get_client_info(self):
        return self.client_info

    async def set_client_info(self, client_info):
        self.client_info = client_info
        self.set_client_info_calls.append(client_info)


@pytest.fixture
def config_path(tmp_path):
    return tmp_path / "mcp_servers.json"


@pytest.fixture
def valid_client_info_dict():
    return {
        "client_id": "clientid",
        "redirect_uris": ["http://localhost:14142/callback"],
    }


@pytest.fixture
def valid_tokens_dict():
    return {
        "access_token": "access_token",
        "token_type": "bearer",
        "refresh_token": "refresh_token",
        "expires_at": int((time.time() + 120) * 1000),
    }


class TestMCPConfig:
    def test_load_config_normalizes_oauth_override(
        self, config_path, valid_tokens_dict, valid_client_info_dict
    ):
        config_data = {
            "server1": {
                "name": "Test Server 1",
                "command": "python",
                "args": ["test_server.py"],
                "env": {"TEST_ENV": "value"},
                "enabledForAgents": ["Engineer"],
                "oauth": {
                    "tokens": valid_tokens_dict,
                    "client_info": valid_client_info_dict,
                },
            },
            "server2": {
                "name": "Test Server 2",
                "command": "node",
                "args": ["test_server.js"],
                "enabledForAgents": [],
            },
        }
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        manager = MCPConfigManager(str(config_path))
        configs = manager.load_config()

        assert len(configs) == 2
        assert configs["server1"].name == "Test Server 1"
        assert configs["server1"].enabledForAgents == ["Engineer"]
        assert configs["server1"].oauth is not None
        assert configs["server1"].oauth.tokens is not None
        assert configs["server1"].oauth.tokens.access_token == "access_token"
        assert configs["server1"].oauth.tokens.refresh_token == "refresh_token"
        assert configs["server1"].oauth.tokens.expires_in is not None
        assert configs["server1"].oauth.tokens.expires_in > 0
        assert configs["server1"].oauth.client_info is not None
        assert configs["server1"].oauth.client_info.client_id == "clientid"
        assert (
            str(configs["server1"].oauth.client_info.redirect_uris[0])
            == "http://localhost:14142/callback"
        )
        assert configs["server2"].oauth is None

    def test_load_config_keeps_valid_oauth_section_when_other_section_invalid(
        self, config_path, valid_client_info_dict
    ):
        config_data = {
            "server1": {
                "name": "Test Server 1",
                "command": "python",
                "args": ["test_server.py"],
                "enabledForAgents": ["Engineer"],
                "oauth": {
                    "tokens": "invalid",
                    "client_info": valid_client_info_dict,
                },
            }
        }
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        manager = MCPConfigManager(str(config_path))
        configs = manager.load_config()

        assert configs["server1"].oauth is not None
        assert configs["server1"].oauth.tokens is None
        assert configs["server1"].oauth.client_info is not None
        assert configs["server1"].oauth.client_info.client_id == "clientid"

    def test_get_enabled_servers_filters_by_enabled_for_agents(self, config_path):
        config_data = {
            "server1": {
                "name": "Test Server 1",
                "command": "python",
                "args": ["test_server.py"],
                "enabledForAgents": ["Engineer"],
            },
            "server2": {
                "name": "Test Server 2",
                "command": "node",
                "args": ["test_server.js"],
                "enabledForAgents": [],
            },
        }
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        manager = MCPConfigManager(str(config_path))
        manager.load_config()

        enabled_servers = manager.get_enabled_servers()
        engineer_servers = manager.get_enabled_servers("Engineer")
        other_servers = manager.get_enabled_servers("Other")

        assert list(enabled_servers.keys()) == ["server1"]
        assert list(engineer_servers.keys()) == ["server1"]
        assert other_servers == {}


@pytest.mark.asyncio
class TestInlineTokenStorage:
    async def test_reads_config_override_before_base_storage(
        self, valid_tokens_dict, valid_client_info_dict
    ):
        base_tokens = OAuthToken.model_validate(
            {
                "access_token": "file_access_token",
                "token_type": "bearer",
                "refresh_token": "file_refresh_token",
                "expires_in": 3600,
            }
        )
        base_client_info = OAuthClientInformationFull.model_validate(
            {
                "client_id": "file-client-id",
                "redirect_uris": ["http://localhost:3000/callback"],
            }
        )
        base_storage = FakeTokenStorage(
            tokens=base_tokens,
            client_info=base_client_info,
        )
        override_storage = InlineTokenStorage(
            base_storage=base_storage,
            tokens_override=OAuthToken.model_validate(
                {
                    **valid_tokens_dict,
                    "expires_in": 1800,
                }
            ),
            client_info_override=OAuthClientInformationFull.model_validate(
                valid_client_info_dict
            ),
        )

        tokens = await override_storage.get_tokens()
        client_info = await override_storage.get_client_info()

        assert tokens.access_token == "access_token"
        assert client_info.client_id == "clientid"

    async def test_falls_back_per_section_when_override_missing(self):
        base_tokens = OAuthToken.model_validate(
            {
                "access_token": "file_access_token",
                "token_type": "bearer",
                "refresh_token": "file_refresh_token",
                "expires_in": 3600,
            }
        )
        base_client_info = OAuthClientInformationFull.model_validate(
            {
                "client_id": "file-client-id",
                "redirect_uris": ["http://localhost:3000/callback"],
            }
        )
        base_storage = FakeTokenStorage(
            tokens=base_tokens,
            client_info=base_client_info,
        )
        override_storage = InlineTokenStorage(
            base_storage=base_storage,
            tokens_override=None,
            client_info_override=OAuthClientInformationFull.model_validate(
                {
                    "client_id": "config-client-id",
                    "redirect_uris": ["http://localhost:14142/callback"],
                }
            ),
        )

        tokens = await override_storage.get_tokens()
        client_info = await override_storage.get_client_info()

        assert tokens.access_token == "file_access_token"
        assert client_info.client_id == "config-client-id"

    async def test_runtime_writes_override_future_reads_and_delegate_persistence(self):
        base_storage = FakeTokenStorage()
        override_storage = InlineTokenStorage(
            base_storage=base_storage,
            tokens_override=OAuthToken.model_validate(
                {
                    "access_token": "initial_access_token",
                    "token_type": "bearer",
                    "refresh_token": "initial_refresh_token",
                    "expires_in": 120,
                }
            ),
            client_info_override=OAuthClientInformationFull.model_validate(
                {
                    "client_id": "initial-client-id",
                    "redirect_uris": ["http://localhost:14142/callback"],
                }
            ),
        )
        new_tokens = OAuthToken.model_validate(
            {
                "access_token": "new_access_token",
                "token_type": "bearer",
                "refresh_token": "new_refresh_token",
                "expires_in": 7200,
            }
        )
        new_client_info = OAuthClientInformationFull.model_validate(
            {
                "client_id": "new-client-id",
                "redirect_uris": ["http://localhost:15151/callback"],
            }
        )

        await override_storage.set_tokens(new_tokens)
        await override_storage.set_client_info(new_client_info)

        assert (await override_storage.get_tokens()).access_token == "new_access_token"
        assert (await override_storage.get_client_info()).client_id == "new-client-id"
        assert base_storage.set_tokens_calls == [new_tokens]
        assert base_storage.set_client_info_calls == [new_client_info]


class TestMCPService:
    def test_build_token_storage_returns_base_storage_without_oauth(self):
        service = MCPService()
        base_storage = FakeTokenStorage()
        server_config = MCPServerConfig(
            name="server1",
            command="python",
            args=["test_server.py"],
            enabledForAgents=["Engineer"],
        )

        service._get_or_create_token_storage = lambda _server_name: base_storage

        token_storage = service._build_token_storage(server_config)

        assert token_storage is base_storage

    def test_build_token_storage_wraps_base_storage_with_oauth_override(
        self, valid_client_info_dict
    ):
        service = MCPService()
        base_storage = FakeTokenStorage()
        server_config = MCPServerConfig(
            name="server1",
            command="python",
            args=["test_server.py"],
            enabledForAgents=["Engineer"],
            oauth=MCPOAuthOverrideConfig(
                client_info=OAuthClientInformationFull.model_validate(
                    valid_client_info_dict
                )
            ),
        )

        service._get_or_create_token_storage = lambda _server_name: base_storage

        token_storage = service._build_token_storage(server_config)

        assert isinstance(token_storage, InlineTokenStorage)
        assert token_storage.base_storage is base_storage
