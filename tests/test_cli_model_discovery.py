"""Official catalog subprocesses, exact model selection and honest metadata."""

import json
import textwrap

import pytest

from flyto_ai import AgentConfig
from flyto_ai.cli_runtime import CliAgent, CliRuntimeConfig, discover_cli_models, complete_json
from flyto_ai.cli_runtime.codex import DISABLED_FEATURES
from flyto_ai.cli_runtime.events import failure_code
from test_cli_runtime import binary


@pytest.mark.asyncio
async def test_codex_catalog_pages_are_dynamic_without_threads_or_turns(tmp_path):
    executable = tmp_path / "catalog"
    log = tmp_path / "methods.jsonl"
    executable.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(f'''
        import json,sys
        for line in sys.stdin:
            event=json.loads(line); method=event['method']
            with open({str(log)!r},'a') as f: f.write(json.dumps(method)+'\\n')
            if method=='initialized': continue
            if method=='initialize': result={{}}
            elif method=='config/read':
                result={{'config':{{'model_provider':'openai','features':{dict.fromkeys(DISABLED_FEATURES,False)!r},
                    'web_search':'disabled','otel':{{'log_user_prompt':False,'exporter':'none','trace_exporter':'none','metrics_exporter':'none'}}}}}}
            elif method=='model/list':
                page=bool(event['params'].get('cursor'))
                result={{'data':[{{'id':'catalog-entry','model':'official-dynamic-model-'+str(page),
                    'displayName':'Current official model','description':'From installed catalog',
                    'isDefault':not page,'inputModalities':['text','image']}}],
                    'nextCursor':None if page else 'next'}}
            else: raise AssertionError('Catalog must not create a thread, start tools or infer')
            print(json.dumps({{'id':event['id'],'result':result}}),flush=True)
    '''))
    executable.chmod(0o700)
    result = await discover_cli_models(CliRuntimeConfig("codex_cli", command=str(executable), model="unavailable-selection"))
    assert result["catalog_available"] and result["manual_entry"]
    assert [row["id"] for row in result["models"]] == ["official-dynamic-model-False", "official-dynamic-model-True"]
    assert result["models"][0]["input_modalities"] == ["text", "image"]
    assert [json.loads(line) for line in log.read_text().splitlines()] == ["initialize", "initialized", "config/read", "model/list", "model/list"]


@pytest.mark.asyncio
async def test_claude_catalog_never_invents_a_list_or_executes_a_binary(tmp_path):
    result = await discover_cli_models(CliRuntimeConfig("claude_cli", command=str(tmp_path / "missing")))
    assert result == {"source": "claude_cli", "models": [], "manual_entry": True,
                      "catalog_available": False, "reason_code": "manual_catalog_unavailable"}


@pytest.mark.asyncio
async def test_manual_official_alias_passes_verbatim_to_actual_cli_argv(tmp_path):
    executable = binary(tmp_path, '''
        assert sys.argv[sys.argv.index('--model')+1]=='sonnet[1m]'
        assert '--fallback-model' not in sys.argv
        finish({'ok':True})
    ''')
    assert json.loads(await complete_json(CliRuntimeConfig("claude_cli", model="sonnet[1m]", command=executable),
                                         prompt="Return JSON.", schema={"type": "object"})) == {"ok": True}


@pytest.mark.parametrize("model", ["--tools=all", "model\nflag", "x y", "x;cmd", ""])
def test_unsafe_cli_model_values_never_become_arguments(model):
    if model == "":
        assert CliRuntimeConfig("codex_cli", model=model).model == ""
    else:
        with pytest.raises(ValueError):
            CliRuntimeConfig("claude_cli", model=model)


@pytest.mark.parametrize("source", ["codex_cli", "claude_cli", "local_ai"])
def test_external_default_does_not_claim_an_openai_model(source):
    assert AgentConfig(provider=source, model="").resolved_model == ""
    assert AgentConfig(provider=source, model="chosen-model").resolved_model == "chosen-model"
    assert AgentConfig(provider="openai", model="").resolved_model == "gpt-4o"


@pytest.mark.asyncio
async def test_delegated_cli_response_keeps_unknown_default_model_empty():
    async def infer(**kwargs):
        return '{"content":"Hello","tool_calls":[]}'
    agent = CliAgent(AgentConfig(enable_transcript=False), cli=CliRuntimeConfig("codex_cli"), completion_fn=infer, tools=[])
    agent._assistant = None
    try:
        result = await agent.chat("Hello")
        assert result.model == "" and result.provider == "codex_cli"
        assert agent.config.resolved_model == ""
    finally:
        await agent.close()


@pytest.mark.parametrize("message", ["model_not_found", "Invalid model requested", "This model is not supported", "The model `selected-name` does not exist or you do not have access to it."])
def test_invalid_selected_model_has_actionable_safe_error(message):
    assert failure_code(message) == "cli_model_unavailable"
