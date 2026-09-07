"""Official app-server inference without an execution environment or MCP tools."""

import json

from .contracts import CliRuntimeError, decode_json, encode_json
from .events import failure_code

DISABLED_FEATURES = (
    'hooks', 'plugins', 'apps', 'browser_use', 'browser_use_external',
    'browser_use_full_cdp_access', 'computer_use', 'image_generation',
    'multi_agent', 'multi_agent_v2', 'shell_tool', 'view_image', 'shell_snapshot',
    'skill_search', 'skill_mcp_dependency_install', 'code_mode', 'code_mode_host',
    'code_mode_only', 'memories', 'goals', 'workspace_dependencies', 'tool_suggest',
    'remote_plugin', 'realtime_conversation', 'sleep_tool', 'token_budget',
)
FIXED_CONFIG = {
    'web_search': 'disabled', 'model_provider': 'openai',
    'approval_policy': 'never', 'sandbox_mode': 'read-only',
    'tools.update_plan.enabled': False,
    'tools.experimental_request_user_input.enabled': False,
    'analytics.enabled': False, 'notify': [], 'project_doc_max_bytes': 0,
    'otel.log_user_prompt': False, 'otel.exporter': 'none',
    'otel.trace_exporter': 'none', 'otel.metrics_exporter': 'none',
}


def codex_argv():
    argv = ['app-server', '--stdio', '--strict-config']
    for key, value in {**FIXED_CONFIG, **{'features.' + key: False for key in DISABLED_FEATURES}}.items():
        argv.extend(('-c', key + '=' + json.dumps(value, separators=(',', ':'))))
    return argv


class CodexProtocol:
    """Fail closed before inference if the installed server cannot isolate tools."""

    def __init__(self, cli, cwd, prompt=None, schema=None, images=()):
        self.cli, self.cwd, self.prompt, self.schema = cli, str(cwd), prompt, schema
        self.images = images
        self.writer = None
        self.thread_id = self.turn_id = ''
        self.model = ''
        self.content = None
        self.usage = {}
        self.completed = False
        self._expected = 1
        self._bytes = 0
        self.initial = self._wire(1, 'initialize', {
            'clientInfo': {'name': 'flyto_cli_runtime', 'version': '1'},
            'capabilities': {'experimentalApi': True},
        })

    @staticmethod
    def _wire(identity, method, params):
        return (encode_json({'id': identity, 'method': method, 'params': params}) + '\n').encode()

    async def _send(self, identity, method, params):
        self._expected = identity
        self.writer.write(self._wire(identity, method, params))
        await self.writer.drain()

    async def read(self, raw):
        event = decode_json(raw)
        if not isinstance(event, dict):
            raise CliRuntimeError('cli_invalid_output')
        if self.completed:
            return  # Shutdown-only notifications cannot alter the sealed result.
        if 'id' in event:
            if 'method' in event:
                raise CliRuntimeError('cli_native_action_refused')
            if event['id'] != self._expected:
                raise CliRuntimeError('cli_invalid_output')
            if 'error' in event:
                raise CliRuntimeError(failure_code(event['error']))
            await self._response(event['id'], event.get('result'))
        else:
            self._notification(event.get('method'), event.get('params') or {})

    async def _response(self, identity, result):
        if not isinstance(result, dict):
            raise CliRuntimeError('cli_invalid_output')
        if identity == 1:
            self.writer.write(b'{"method":"initialized"}\n')
            await self._send(2, 'config/read', {'includeLayers': False})
        elif identity == 2:
            config = result.get('config') or {}
            self._check_config(config)
            servers = config.get('mcp_servers', {})
            if not isinstance(servers, dict) or len(servers) > 100:
                raise CliRuntimeError('cli_configuration_unsupported')
            overrides = {'mcp_servers': {name: {'enabled': False} for name in servers}}
            params = {
                'cwd': self.cwd, 'environments': [], 'selectedCapabilityRoots': [],
                'dynamicTools': [], 'ephemeral': True, 'approvalPolicy': 'never',
                'sandbox': 'read-only', 'modelProvider': 'openai', 'config': overrides,
                'baseInstructions': 'Return the requested structured JSON. You have no tools or execution environment.',
                'developerInstructions': 'Only the external host may execute actions. Do not invoke native tools.',
            }
            if self.cli.model:
                params['model'] = self.cli.model
            await self._send(3, 'thread/start', params)
        elif identity == 3:
            thread = result.get('thread') or {}
            if (thread.get('ephemeral') is not True or result.get('modelProvider') != 'openai'
                    or result.get('approvalPolicy') != 'never'
                    or result.get('sandbox', {}).get('type') != 'readOnly'):
                raise CliRuntimeError('cli_tool_isolation_unavailable')
            self.thread_id = thread.get('id')
            if not isinstance(self.thread_id, str) or not self.thread_id:
                raise CliRuntimeError('cli_invalid_output')
            self.model = result.get('model', '')
            await self._send(4, 'mcpServerStatus/list', {'threadId': self.thread_id, 'limit': 100})
        elif identity == 4:
            servers = result.get('data')
            if not isinstance(servers, list) or result.get('nextCursor') or any(
                not isinstance(row, dict) or row.get('runtimeStatus') != 'disabled'
                or row.get('tools') or row.get('resources') or row.get('resourceTemplates')
                for row in servers
            ):
                raise CliRuntimeError('cli_native_tools_exposed')
            if self.prompt is None:
                self.content = {'supported': True}
                self._finish()
                return
            content = [{'type': 'text', 'text': self.prompt, 'text_elements': []}]
            content.extend({'type': 'image', 'url': 'data:' + row['media_type'] + ';base64,' + row['base64']} for row in self.images)
            await self._send(5, 'turn/start', {
                'threadId': self.thread_id, 'environments': [], 'input': content,
                'outputSchema': self.schema,
            })
        elif identity == 5:
            self.turn_id = (result.get('turn') or {}).get('id', '')

    @staticmethod
    def _check_config(config):
        if config.get('model_provider') != 'openai' or any(
            config.get('features', {}).get(key) is not False for key in DISABLED_FEATURES
        ):
            raise CliRuntimeError('cli_tool_isolation_unavailable')
        # Keep official ChatGPT sign-in routing. Forcing an API base URL breaks
        # its credentials; inheriting a custom endpoint would leak host inputs.
        if config.get('openai_base_url') or config.get('chatgpt_base_url') not in (
            None, 'https://chatgpt.com/backend-api', 'https://chatgpt.com/backend-api/',
        ):
            raise CliRuntimeError('cli_nondefault_provider_route')
        if config.get('notify') or config.get('web_search') != 'disabled':
            raise CliRuntimeError('cli_tool_isolation_unavailable')
        telemetry = config.get('otel') or {}
        if telemetry.get('log_user_prompt') is not False or any(
            telemetry.get(key) != 'none' for key in ('exporter', 'trace_exporter', 'metrics_exporter')
        ):
            raise CliRuntimeError('cli_configuration_unsupported')

    def _notification(self, method, params):
        if params.get('threadId') and self.thread_id and params['threadId'] != self.thread_id:
            raise CliRuntimeError('cli_session_changed')
        if method in ('item/started', 'item/completed'):
            item = params.get('item') or {}
            if item.get('type') not in ('userMessage', 'reasoning', 'agentMessage'):
                raise CliRuntimeError('cli_native_action_refused')
            if method == 'item/completed' and item.get('type') == 'agentMessage':
                self.content = decode_json(item.get('text', ''))
        elif method == 'turn/completed':
            turn = params.get('turn') or {}
            if turn.get('status') != 'completed':
                raise CliRuntimeError(failure_code(turn.get('error')))
            if self.turn_id and turn.get('id') != self.turn_id:
                raise CliRuntimeError('cli_session_changed')
            self._finish()
        elif method == 'thread/tokenUsage/updated':
            usage = (params.get('tokenUsage') or {}).get('last', {})
            for key, name in (('inputTokens', 'prompt_tokens'), ('outputTokens', 'completion_tokens'), ('totalTokens', 'total_tokens')):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9:
                    self.usage[name] = value
        elif method == 'error':
            raise CliRuntimeError(failure_code(params))
        elif method and method.startswith(('hook/', 'item/tool', 'item/command')):
            raise CliRuntimeError('cli_native_action_refused')

    def _finish(self):
        self.completed = True
        self.writer.close()

    def result(self):
        if not self.completed or self.content is None:
            raise CliRuntimeError('cli_incomplete_output')
        return self.content, self.usage
