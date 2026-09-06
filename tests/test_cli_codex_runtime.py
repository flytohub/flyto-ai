"""App-server fixtures execute the real SDK subprocess protocol, never a model."""

import json
import textwrap

import pytest

from flyto_ai.cli_runtime import (
    CliRuntimeConfig,
    CliRuntimeError,
    complete_json,
    inspect_cli_runtime,
)
from flyto_ai.cli_runtime.codex import DISABLED_FEATURES, CodexProtocol
from flyto_ai.cli_runtime.process import ProcessRunner


def codex_binary(tmp_path, *, scenario='normal'):
    path = tmp_path / 'codex-fixture'
    path.write_text('#!/usr/bin/env python3\n' + textwrap.dedent(f'''
        import sys,json
        if '--version' in sys.argv:
            print('codex-cli 0.153.4');sys.exit()
        if '--help' in sys.argv:
            print('--stdio --strict-config --listen');sys.exit()
        assert sys.argv[1:4]==['app-server','--stdio','--strict-config']
        assert 'notify=[]' in sys.argv
        assert 'openai_base_url=' not in ' '.join(sys.argv)
        scenario={scenario!r}
        def emit(value):print(json.dumps(value),flush=True)
        for raw in sys.stdin:
            event=json.loads(raw);method=event['method'];i=event.get('id')
            if method=='initialized':continue
            p=event['params']
            if method=='initialize':result={{}}
            elif method=='config/read':
                result={{'config':{{'model_provider':'openai','features':{dict.fromkeys(DISABLED_FEATURES,False)!r},'web_search':'disabled','notify':[],
                'otel':{{'log_user_prompt':False,'exporter':'none','trace_exporter':'none','metrics_exporter':'none'}},
                'mcp_servers':{{'private.server':{{'command':'NEVER EXECUTE THIS','env':{{'SECRET':'never-print'}}}}}},'chatgpt_base_url':'https://chatgpt.com/backend-api/'}}}}
                if scenario=='foreign_route':result['config']['openai_base_url']='https://not-the-selected-provider.invalid'
            elif method=='thread/start':
                assert p['environments']==[] and p['dynamicTools']==[] and p['selectedCapabilityRoots']==[]
                assert p['config']=={{'mcp_servers':{{'private.server':{{'enabled':False}}}}}}
                assert p['ephemeral'] is True and p['approvalPolicy']=='never' and p['sandbox']=='read-only'
                result={{'thread':{{'id':'fixture-thread','ephemeral':True}},'modelProvider':'openai','model':'actual-cli-model','approvalPolicy':'never','sandbox':{{'type':'readOnly','networkAccess':False}}}}
            elif method=='mcpServerStatus/list':
                result={{'data':[{{'runtimeStatus':'disabled','tools':{{}},'resources':[],'resourceTemplates':[]}}],'nextCursor':None}}
                if scenario=='mcp_exposed':result['data'][0]['tools']={{'hidden':{{}}}}
                if scenario=='mcp_loading':result['data'][0]['runtimeStatus']='connecting'
            elif method=='turn/start':
                assert p['environments']==[] and p['outputSchema']['type']=='object'
                if scenario=='native_request':
                    emit({{'id':333,'method':'item/commandExecution/requestApproval','params':{{}}}});continue
                result={{'turn':{{'id':'fixture-turn'}}}}
                emit({{'id':i,'result':result}})
                if scenario=='native_action':
                    emit({{'method':'item/started','params':{{'threadId':'fixture-thread','item':{{'type':'commandExecution'}}}}}});continue
                emit({{'method':'item/completed','params':{{'threadId':'fixture-thread','item':{{'type':'agentMessage','text':'{{"ok":true}}'}}}}}})
                emit({{'method':'thread/tokenUsage/updated','params':{{'threadId':'fixture-thread','tokenUsage':{{'last':{{'inputTokens':7,'outputTokens':3,'totalTokens':10}}}}}}}})
                emit({{'method':'turn/completed','params':{{'threadId':'fixture-thread','turn':{{'id':'fixture-turn','status':'completed'}}}}}})
                continue
            else:raise AssertionError('Unexpected protocol request')
            emit({{'id':i,'result':result}})
    '''))
    path.chmod(0o700)
    return str(path)


@pytest.mark.asyncio
async def test_codex_official_protocol_disables_mcp_and_environment_before_turn(tmp_path):
    cli=CliRuntimeConfig('codex_cli',command=codex_binary(tmp_path))
    status=await inspect_cli_runtime(cli)
    assert status == {'source':'codex_cli','installed':True,'version':'0.153.4','supported':True,'reason_code':'ready'}
    runner=ProcessRunner(cli)
    try:
        result,usage=await runner.infer('Return only structured JSON.',{'type':'object'})
    finally:
        await runner.close()
    assert result == {'ok':True} and usage['total_tokens']==10
    assert runner.last_model == 'actual-cli-model' and not runner._processes


@pytest.mark.asyncio
@pytest.mark.parametrize(('scenario','error'),[
    ('foreign_route','cli_nondefault_provider_route'),
    ('mcp_exposed','cli_native_tools_exposed'),
    ('mcp_loading','cli_native_tools_exposed'),
    ('native_request','cli_native_action_refused'),
    ('native_action','cli_native_action_refused'),
])
async def test_codex_cannot_inherit_or_invoke_unselected_actions(tmp_path,scenario,error):
    with pytest.raises(CliRuntimeError,match=error):
        await complete_json(CliRuntimeConfig('codex_cli',command=codex_binary(tmp_path,scenario=scenario)),
                            prompt='Return JSON only.',schema={'type':'object'})


def test_codex_environment_binding_and_native_actions_cannot_arrive_from_another_thread():
    protocol=CodexProtocol(CliRuntimeConfig('codex_cli'),'/tmp')
    protocol.thread_id='owned'
    with pytest.raises(CliRuntimeError,match='cli_session_changed'):
        protocol._notification('item/completed',{'threadId':'other','item':{'type':'agentMessage','text':'{}'}})


def test_claude_structured_output_is_not_an_execution_capability():
    from flyto_ai.cli_runtime.events import EventReader
    reader=EventReader('claude_cli')
    reader.read(json.dumps({'type':'system','subtype':'init','tools':['StructuredOutput']}).encode())
    with pytest.raises(CliRuntimeError,match='cli_native_tools_exposed'):
        reader.read(json.dumps({'type':'system','subtype':'init','tools':['StructuredOutput','Bash']}).encode())


def test_claude_formatter_acknowledgement_is_bound_to_observed_call_id():
    from flyto_ai.cli_runtime.events import EventReader
    reader=EventReader('claude_cli')
    reader.read(json.dumps({'type':'assistant','message':{'content':[{'type':'tool_use','name':'StructuredOutput','id':'formatter-1'}]}}).encode())
    reader.read(json.dumps({'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'formatter-1','content':'accepted'}]}}).encode())
    with pytest.raises(CliRuntimeError,match='cli_native_action_refused'):
        reader.read(json.dumps({'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'foreign-execution','content':'claimed success'}]}}).encode())


@pytest.mark.asyncio
async def test_complete_json_byte_limit_is_not_character_count(tmp_path):
    cli=CliRuntimeConfig('codex_cli',command=codex_binary(tmp_path))
    with pytest.raises(CliRuntimeError,match='cli_invalid_input'):
        await complete_json(cli,prompt='讀'*100000,schema={'type':'object'})
    assert json.loads(await complete_json(cli,prompt='a'*60000,schema={'type':'object'}))=={'ok':True}
