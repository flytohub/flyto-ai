# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Agent class — chat loop orchestrator.

The Agent is a thin shell: config → provider → tools → system prompt → chat loop.
All assistant intelligence (blueprint routing, interactive input, selector healing)
lives in ``flyto_ai.assistant.AssistantMiddleware``.
"""
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.closed_loop_v3 import (
    CapabilityModelRouter,
    JsonCheckpointStore,
    ModelCandidate,
    ModelRoute,
)
from flyto_ai.config import AgentConfig
from flyto_ai.intelligence.confirmation import ToolIntentDecision, classify_tool_intent, route_with_confirmation
from flyto_ai.models import ChatResponse, StreamCallback, StreamEvent, StreamEventType, UsageStats
from flyto_ai.permissions import TOOL_PERMISSION_MAP, PermissionEnforcer, PermissionLevel
from flyto_ai.prompt.policies import is_module_allowed, is_tool_allowed
from flyto_ai.prompt.system_prompt import build_system_prompt, detect_language
from flyto_ai.protocols import ApiClient, ToolExecutor
from flyto_ai.providers.base import LLMProvider
from flyto_ai.validation import extract_yaml_from_response, validate_workflow_steps

logger = logging.getLogger(__name__)

# Lazy imports for optional modules
_CostTracker = None
_TranscriptWriter = None
_injection_detector = None
_Vault = None


def _merge_usage(accumulated: Dict[str, int], new: UsageStats) -> None:
    """Merge new usage stats into accumulated dict (in-place)."""
    accumulated["prompt_tokens"] += new.prompt_tokens
    accumulated["completion_tokens"] += new.completion_tokens
    accumulated["total_tokens"] += new.total_tokens
    accumulated["cache_creation_input_tokens"] += new.cache_creation_input_tokens
    accumulated["cache_read_input_tokens"] += new.cache_read_input_tokens


# Phrases a model uses when it presents a run as under way or about to start.
# Checked only in execute mode, only against a reply that called nothing that
# runs. The owner's turn read 「我將執行 "kintone" 工作流程來幫助您登入。請稍候。
# 執行中...」 with tool_calls_count=0; the chat showed progress for a run that
# never began until the user typed 執行啊.
_COMMITMENT_PHRASES = (
    "執行中", "执行中", "處理中", "处理中", "請稍候", "请稍候", "請稍等", "请稍等",
    "稍等", "我將", "我将", "我會執行", "我会执行", "為您執行", "为您执行",
    "正在執行", "正在执行", "正在為", "正在为", "馬上", "马上", "立即執行",
    "立即执行", "開始執行", "开始执行",
    "実行します", "実行中", "お待ちください",
    "실행하겠습니다", "실행 중", "잠시만",
    "i will ", "i'll ", "i am going to", "i'm going to", "let me ", "executing",
    "is running", "now running", "running the", "please wait", "one moment",
    "hold on", "in progress", "kicking off", "starting the",
)


def _reads_as_commitment(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _COMMITMENT_PHRASES)


def _nothing_ran_message(language: str, workflow_names: List[str]) -> str:
    """The reply for a turn that promised a run and made no call.

    Must not itself contain any of the phrases above -- the whole point is
    that no fake progress word reaches the user.
    """
    if language.startswith("Traditional Chinese"):
        if workflow_names:
            return "我沒有執行任何工作流程。這裡可以執行的工作流程：{}。要我現在執行它嗎？".format(
                "、".join(workflow_names),
            )
        return "我沒有執行任何操作，也沒有任何工作流程在跑。請告訴我要執行哪一個。"
    if language.startswith("Simplified Chinese"):
        if workflow_names:
            return "我没有执行任何工作流程。这里可以执行的工作流程：{}。要我现在执行它吗？".format(
                "、".join(workflow_names),
            )
        return "我没有执行任何操作，也没有任何工作流程在跑。请告诉我要执行哪一个。"
    if language.startswith("Japanese"):
        if workflow_names:
            return "ワークフローはまだ実行していません。ここで実行できるワークフロー：{}。今すぐ実行しますか？".format(
                "、".join(workflow_names),
            )
        return "まだ何も実行していません。どのワークフローを実行するか教えてください。"
    if workflow_names:
        return "Nothing was run. The workflow available here is {}. Ask me to run it and it gets called as a tool.".format(
            ", ".join(workflow_names),
        )
    return "Nothing was run: no module or workflow was executed this turn. Tell me exactly what to run."


def _bind_tool_executor(
    tool_executor: Optional[ToolExecutor],
    tools: Optional[List[Dict]],
    dispatch_fn,
):
    """Bind one generic executor and validate optional permission metadata."""
    if tool_executor is None:
        return tools or [], dispatch_fn, {}
    bound_tools = tool_executor.tools
    bound_dispatch = tool_executor.dispatch
    if not isinstance(bound_tools, list):
        raise ValueError("tool_executor tools must be a list")
    if not callable(bound_dispatch):
        raise ValueError("tool_executor dispatch must be callable")
    declared_overrides = getattr(tool_executor, "permission_overrides", {})
    if not isinstance(declared_overrides, dict) or any(
        not isinstance(name, str) or not isinstance(level, PermissionLevel)
        for name, level in declared_overrides.items()
    ):
        raise ValueError(
            "tool_executor permission_overrides must map names to PermissionLevel",
        )
    return bound_tools, bound_dispatch, dict(declared_overrides)


class Agent:
    """High-level AI agent that translates natural language to Flyto2 workflows.

    Wires together: config → provider → tools → assistant → system prompt → chat loop.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List[Dict]] = None,
        dispatch_fn=None,
        system_prompt: Optional[str] = None,
        policies: Optional[Dict[str, Any]] = None,
        *,
        api_client: Optional[ApiClient] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ) -> None:
        self._config = config
        self._provider = api_client or self._make_provider()
        self._model_router = CapabilityModelRouter()
        self._model_candidates = self._build_model_candidates()
        self._last_model_route: Optional[ModelRoute] = None
        self._checkpoint_store = None
        if config.enable_checkpoints:
            try:
                self._checkpoint_store = JsonCheckpointStore(
                    config.checkpoint_dir,
                )
            except Exception as exc:
                logger.warning("Checkpoint store init failed: %s", exc)
        self._system_prompt = system_prompt
        self._policies = policies
        self._tool_executor = tool_executor

        # When a ToolExecutor is provided, derive tools + dispatch from it.
        self._tools, self._dispatch_fn, permission_overrides = _bind_tool_executor(
            tool_executor, tools, dispatch_fn,
        )

        # Permission enforcer (three-tier: READ_ONLY / WORKSPACE_WRITE / DANGER_FULL)
        self._permission_enforcer = PermissionEnforcer(
            level=PermissionLevel[config.permission_level.upper()],
            overrides=permission_overrides,
        )
        self._preferred_language: Optional[str] = None
        self._last_routing_decision: Optional[ToolIntentDecision] = None
        self._routing_metrics = {
            "turns": 0,
            "answer_only_turns": 0,
            "ambiguous_turns": 0,
            "action_turns": 0,
            "tool_calls_attempted": 0,
            "tool_calls_executed": 0,
            "tool_calls_blocked": 0,
        }

        # Memory system (lazy init)
        self._memory_store = None
        self._summarizer = None
        self._memory_search = None
        self._memory_initialized = False
        self._closed = False
        self._session_id = uuid.uuid4().hex[:12]

        # Assistant middleware — single entry point for all intelligence
        self._assistant = self._init_assistant()

        # Phase 1: Cost tracker
        self._cost_tracker = self._init_cost_tracker()

        # Phase 1: Transcript writer
        self._transcript = self._init_transcript()

        # Phase 1: Vault (credential store)
        self._vault = self._init_vault()

        # Phase 1.5: flyto-pro intelligence bridge
        self._pro = self._init_pro_bridge()

        # Phase 2: Context compactor
        self._compactor = self._init_compactor()

        # Phase 2: Extension hooks
        self._hooks = self._init_extensions()

        # Phase 2: Orchestrator (lazy — only created when sub-agents spawn)
        self._orchestrator = None

        # Sandbox
        if config.enable_sandbox:
            self._init_sandbox()

        # Auto-discover tools when nothing injected
        if not self._tools and not self._dispatch_fn:
            self._auto_discover_tools()

    # ── Properties ────────────────────────────────────────────────

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def tools(self) -> List[Dict]:
        return list(self._tools) if self._tools else []

    @property
    def dispatch_fn(self):
        return self._dispatch_fn

    @property
    def tool_executor(self) -> Optional[ToolExecutor]:
        """Return the injected executor so hosts can inspect its evidence APIs."""
        return self._tool_executor

    @property
    def memory_store(self):
        return self._memory_store

    @property
    def memory_search(self):
        return self._memory_search

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def cost_tracker(self):
        return self._cost_tracker

    @property
    def pro(self):
        """flyto-pro intelligence bridge (None if unavailable)."""
        return self._pro

    @property
    def model_route(self) -> Optional[Dict[str, Any]]:
        """Return the most recent auditable model-routing decision."""
        if self._last_model_route is None:
            return None
        return self._last_model_route.to_dict()

    async def __aenter__(self) -> "Agent":
        if self._closed:
            raise RuntimeError("agent is closed")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        await self.close()

    async def close(self) -> None:
        """Close agent-owned persistence and transcript resources once."""
        if self._closed:
            return
        self._closed = True
        memory_store = self._memory_store
        transcript = self._transcript
        self._memory_store = None
        self._memory_search = None
        self._summarizer = None
        self._transcript = None
        try:
            if memory_store is not None:
                await memory_store.close()
        finally:
            if transcript is not None:
                transcript.close()

    @property
    def routing_decision(self) -> Optional[Dict[str, Any]]:
        """Return the latest deterministic conversation-routing evidence."""
        self._ensure_routing_state()
        if self._last_routing_decision is None:
            return None
        return self._last_routing_decision.to_dict()

    @property
    def routing_metrics(self) -> Dict[str, int]:
        """Return counters suitable for false-activation and tool-use evals."""
        self._ensure_routing_state()
        return dict(self._routing_metrics)

    @property
    def transcript(self):
        return self._transcript

    @property
    def vault(self):
        return self._vault

    # ── Init helpers ──────────────────────────────────────────────

    def _init_assistant(self):
        """Initialize the AssistantMiddleware (blueprint, interactive, resilience)."""
        try:
            from flyto_ai.assistant import AssistantMiddleware
            return AssistantMiddleware(
                distillation_min_steps=self._config.distillation_min_steps,
            )
        except Exception as e:
            logger.debug("Assistant middleware init failed: %s", e)
            return None

    def _init_cost_tracker(self):
        try:
            from flyto_ai.cost import CostTracker
            return CostTracker(
                session_budget_usd=self._config.session_budget_usd,
                global_budget_usd=self._config.global_budget_usd,
            )
        except Exception as e:
            logger.debug("Cost tracker init failed: %s", e)
            return None

    def _init_transcript(self):
        if not self._config.enable_transcript:
            return None
        try:
            from flyto_ai.transcript import TranscriptWriter
            tw = TranscriptWriter(
                session_id=self._session_id,
                transcript_dir=self._config.transcript_dir,
            )
            tw.record_meta({
                "event": "session_start",
                "provider": self._config.provider,
                "model": self._config.resolved_model,
            })
            return tw
        except Exception as e:
            logger.debug("Transcript init failed: %s", e)
            return None

    def _init_vault(self):
        try:
            from flyto_ai.vault import Vault
            vault = Vault(
                vault_path=self._config.vault_path,
                passphrase=self._config.vault_passphrase,
            )
            vault.load()
            if self._config.vault_auto_inject:
                vault.inject_to_env()
            return vault
        except ImportError:
            logger.debug("Vault unavailable (cryptography not installed)")
            return None
        except Exception as e:
            logger.debug("Vault init failed: %s", e)
            return None

    def _init_pro_bridge(self):
        """Initialize the flyto-pro intelligence bridge."""
        if not self._config.enable_pro:
            return None
        try:
            from flyto_ai.intelligence.pro_bridge import ProBridge
            bridge = ProBridge(config=self._config)
            if bridge.available:
                # Share EMS with assistant middleware for error learning
                if self._assistant and self._config.enable_ems:
                    self._assistant._ems_bridge = bridge
                return bridge
            return None
        except Exception as e:
            logger.debug("Pro bridge init failed: %s", e)
            return None

    def _init_compactor(self):
        try:
            from flyto_ai.memory.compaction import ContextCompactor
            return ContextCompactor()
        except Exception as e:
            logger.debug("Compactor init failed: %s", e)
            return None

    def _init_extensions(self):
        try:
            from flyto_ai.extensions.loader import ExtensionLoader
            loader = ExtensionLoader()
            registry = loader.load_all(
                allowed_capabilities={"read_messages", "read_tool_results"},
            )
            if registry.extension_count > 0:
                logger.info("Loaded %d extensions", registry.extension_count)
            return registry
        except Exception as e:
            logger.debug("Extensions init failed: %s", e)
            return None

    def get_orchestrator(self):
        if self._orchestrator is None:
            from flyto_ai.orchestration import AgentOrchestrator
            self._orchestrator = AgentOrchestrator(
                parent_session_id=self._session_id,
                config=self._config,
            )
        return self._orchestrator

    def _auto_discover_tools(self):
        """Auto-detect and register available tools."""
        from flyto_ai.tools.registry import ToolRegistry

        registry = ToolRegistry()

        # Blueprint tools FIRST — LLMs prefer tools listed earlier
        try:
            from flyto_ai.tools.blueprint_tools import get_blueprint_tool_defs, dispatch_blueprint_tool
            defs = get_blueprint_tool_defs()
            if defs:
                registry.register_many(defs, dispatch_blueprint_tool)
        except Exception as e:
            logger.warning("Failed to load blueprint tools: %s", e)

        try:
            from flyto_ai.tools.core_tools import get_core_tool_defs, dispatch_core_tool
            defs = get_core_tool_defs()
            if defs:
                registry.register_many(defs, dispatch_core_tool)
        except Exception as e:
            logger.warning("Failed to load core tools: %s", e)

        try:
            from flyto_ai.tools.ask_user import TOOL_DEF as ASK_USER_TOOL, dispatch_ask_user
            registry.register(ASK_USER_TOOL, dispatch_ask_user)
        except Exception as e:
            logger.warning("Failed to load ask_user tool: %s", e)

        try:
            from flyto_ai.tools.navigator import TOOL_DEF as NAV_TOOL, dispatch_navigator
            registry.register(NAV_TOOL, dispatch_navigator)
        except Exception as e:
            logger.warning("Failed to load navigator tool: %s", e)

        try:
            from flyto_ai.tools.inspect_page import INSPECT_PAGE_TOOL, dispatch_inspect_page
            registry.register(INSPECT_PAGE_TOOL, dispatch_inspect_page)
        except Exception as e:
            logger.warning("Failed to load inspect_page tool: %s", e)

        if registry.tools:
            self._tools = registry.tools
            self._dispatch_fn = registry.dispatch
            # Set dispatch ref for navigator
            try:
                from flyto_ai.tools.navigator import set_dispatch
                set_dispatch(registry.dispatch)
            except Exception:
                pass

    def _init_sandbox(self):
        try:
            from flyto_ai.sandbox.manager import SandboxManager
            from flyto_ai.tools.core_tools import set_sandbox_manager
            mgr = SandboxManager(
                image=self._config.sandbox_image,
                timeout=self._config.sandbox_timeout,
            )
            set_sandbox_manager(mgr)
            logger.info("Sandbox enabled: image=%s", self._config.sandbox_image)
        except Exception as e:
            logger.warning("Failed to init sandbox: %s", e)

    async def _init_memory(self):
        if self._memory_initialized:
            return
        self._memory_initialized = True

        if not self._config.enable_memory:
            return

        try:
            from flyto_ai.memory.sqlite_store import SQLiteSessionStore
            from flyto_ai.memory.summarizer import ConversationSummarizer

            self._memory_store = SQLiteSessionStore(db_path=self._config.memory_db_path)
            await self._memory_store.init()
            self._summarizer = ConversationSummarizer(
                provider=self._provider, threshold=20, keep_recent=10,
            )

            try:
                from flyto_ai.memory.embeddings import EmbeddingStore
                from flyto_ai.memory.bm25 import BM25Index
                from flyto_ai.memory.search import MemorySearch

                db = self._memory_store._db
                # Embeddings go over the OpenAI wire protocol, so hand them the
                # same key and base_url the chat provider was built with (same
                # branching as _make_provider). An Anthropic key is no use to
                # /v1/embeddings, so that provider keeps the environment fallback.
                cfg = self._config
                embed_kwargs = {}
                if cfg.provider == "ollama":
                    embed_kwargs["base_url"] = cfg.base_url or "http://localhost:11434/v1"
                elif cfg.provider != "anthropic":
                    embed_kwargs["api_key"] = cfg.api_key
                    embed_kwargs["base_url"] = cfg.base_url
                emb = EmbeddingStore(db, model=cfg.embedding_model, **embed_kwargs)
                await emb.init()
                bm25 = BM25Index(db)
                await bm25.init()
                self._memory_search = MemorySearch(emb, bm25)
            except Exception as e:
                logger.debug("Memory search init failed (BM25-only fallback): %s", e)
        except Exception as e:
            logger.warning("Memory system init failed: %s", e)

    @staticmethod
    def _default_model_for(provider: str) -> str:
        if provider == "anthropic":
            return "claude-sonnet-4-5-20250929"
        if provider == "ollama":
            return "llama3.2"
        return "gpt-4o"

    @staticmethod
    def _model_cost_rank(provider: str, model: str) -> int:
        lowered = model.lower()
        if provider == "ollama":
            return 0
        if "mini" in lowered or "haiku" in lowered:
            return 1
        if "gpt-5" in lowered or "opus" in lowered or "sonnet" in lowered:
            return 3
        return 2

    def _build_model_candidates(self) -> List[ModelCandidate]:
        cfg = self._config
        configured = [{
            "provider": cfg.provider or "openai",
            "model": cfg.resolved_model,
        }]
        configured.extend({
            "provider": item.provider or "openai",
            "model": item.model or self._default_model_for(item.provider),
        } for item in cfg.fallback_providers)

        candidates = []
        seen = set()
        for item in configured:
            label = "{}:{}".format(item["provider"], item["model"])
            if label in seen:
                continue
            seen.add(label)
            candidates.append(ModelCandidate.from_name(
                item["provider"],
                item["model"],
                self._model_cost_rank(item["provider"], item["model"]),
            ))
        return candidates

    def _select_model_route(
        self,
        message: str,
        *,
        deterministic_available: bool = False,
        prior_failure: bool = False,
        plan_steps: int = 0,
    ) -> ModelRoute:
        if not self._config.enable_model_routing and not deterministic_available:
            primary = self._model_candidates[0]
            route = ModelRoute(
                mode="llm",
                required_tier=primary.tier,
                reason="capability routing disabled",
                provider=primary.provider,
                model=primary.model,
                candidate_label=primary.label,
            )
        else:
            route = self._model_router.route(
                message,
                self._model_candidates,
                deterministic_available=deterministic_available,
                prior_failure=prior_failure,
                plan_steps=plan_steps,
            )

        if route.mode == "llm" and route.candidate_label:
            prefer = getattr(self._provider, "prefer_provider", None)
            primary_label = self._model_candidates[0].label
            if callable(prefer):
                applied = bool(prefer(route.candidate_label))
            else:
                applied = route.candidate_label == primary_label
            if not applied:
                primary = self._model_candidates[0]
                route = ModelRoute(
                    mode="llm",
                    required_tier=route.required_tier,
                    reason="{}; selected provider cannot be activated".format(
                        route.reason,
                    ),
                    provider=primary.provider,
                    model=primary.model,
                    candidate_label=primary.label,
                    degraded=True,
                )
        self._last_model_route = route
        return route

    def _make_provider(self) -> LLMProvider:
        from flyto_ai.providers import create_provider

        cfg = self._config
        kwargs = {
            "model": cfg.resolved_model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.provider == "ollama":
            kwargs["base_url"] = cfg.base_url or "http://localhost:11434/v1"
        else:
            kwargs["api_key"] = cfg.api_key
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url

        primary = create_provider(cfg.provider or "openai", **kwargs)

        if cfg.fallback_providers:
            try:
                from flyto_ai.providers.failover import ProviderChain
                fallbacks = []
                names = ["{}:{}".format(cfg.provider or "openai", cfg.resolved_model)]
                for fb in cfg.fallback_providers:
                    fb_kwargs = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
                    if fb.model:
                        fb_kwargs["model"] = fb.model
                    if fb.provider == "ollama":
                        fb_kwargs["base_url"] = fb.base_url or "http://localhost:11434/v1"
                    else:
                        fb_kwargs["api_key"] = fb.api_key
                        if fb.base_url:
                            fb_kwargs["base_url"] = fb.base_url
                    fallbacks.append(create_provider(fb.provider or "openai", **fb_kwargs))
                    names.append("{}:{}".format(fb.provider, fb.model or "default"))
                logger.info("Provider chain: %s", " → ".join(names))
                return ProviderChain(primary=primary, fallbacks=fallbacks, provider_names=names)
            except Exception as e:
                logger.warning("Failed to create provider chain, using primary only: %s", e)

        return primary

    # ── Deterministic pipeline ─────────────────────────────────────

    async def _try_deterministic(
        self, message: str, on_tool_call, on_stream, dispatch_wrapper,
        routing_decision: Optional[ToolIntentDecision] = None,
    ) -> Optional[ChatResponse]:
        """Try to handle the message with deterministic planning (zero LLM).

        Returns a ChatResponse if handled, None to fall back to LLM.
        """
        routing_decision = routing_decision or classify_tool_intent(message)
        if not routing_decision.tool_eligible:
            return None

        try:
            from flyto_ai.intelligence.planner import extract_intent, extract_intent_llm, plan_execution, execute_plan, _resolve_url
            from flyto_ai.tools.core_tools import get_browser_status
        except ImportError:
            return None

        # 1. Extract intent — try data-driven first, then 1 cheap LLM call
        intent = extract_intent(message)
        if intent is None:
            # Try LLM intent extraction (1 call, ~$0.001)
            llm_intent = await extract_intent_llm(message, self._provider)
            if llm_intent is None:
                return None  # Question/conversation → full LLM

            # Map LLM intent to planner format
            action = llm_intent.get("action", "")
            target = llm_intent.get("target", "")
            query = llm_intent.get("query", "")

            if action in ("navigate", "open", "go"):
                url = llm_intent.get("url") or _resolve_url(target)
                if url:
                    if query:
                        intent = {"intent": "open_and_search", "url": url, "search_text": query, "site": target}
                    else:
                        intent = {"intent": "open_website", "url": url, "site": target}
                else:
                    return None
            elif action in ("search", "find"):
                intent = {"intent": "search_on_website", "search_text": query or target}
            elif action in ("click", "play", "select", "tap"):
                intent = {"intent": "click_element", "target": target or query}
            elif action in ("screenshot",):
                intent = {"intent": "single_module", "module_id": "browser.screenshot", "params": {}}
            else:
                # Try as single module with LLM-extracted params
                intent = {"intent": "single_module", "module_id": action + "." + target if "." not in action else action,
                          "params": llm_intent.get("params", {})}

        if intent is None:
            return None

        logger.info("Deterministic intent: %s", intent.get("intent"))
        deterministic_route = None
        if hasattr(self, "_model_router") and hasattr(self, "_model_candidates"):
            deterministic_route = self._select_model_route(
                message,
                deterministic_available=True,
            )

        dispatch_fn, _ = self._build_dispatch(
            message, on_tool_call, on_stream, dispatch_wrapper,
            mode="execute",
            blueprint_selection_mode="deterministic",
            routing_decision=routing_decision,
            active_tools=self._tools_for_route(routing_decision, "execute"),
        )
        if dispatch_fn is None:
            return None

        # Exact blueprint reuse: zero LLM, but never bypass the agent's safety
        # boundary. ``use_blueprint`` expands the pattern and safe_dispatch
        # executes every Core step with validation before reporting outcome.
        if intent.get("intent") == "blueprint":
            blueprint_args = {
                "blueprint_id": intent.get("blueprint_id", ""),
                "args": intent.get("args", {}),
            }
            result = await dispatch_fn("use_blueprint", blueprint_args)
            if not isinstance(result, dict) or not result.get("workflow_executed"):
                return None

            executions = [
                item for item in result.get("executions", [])
                if isinstance(item, dict)
            ]
            evidence = result.get("evidence", {})
            if deterministic_route is not None:
                evidence["model_route"] = deterministic_route.to_dict()
            modules = [
                item.get("module_id", "") for item in executions
                if item.get("module_id")
            ]
            closed_loop_ok = bool(result.get("closed_loop_ok"))
            if closed_loop_ok:
                response_text = "OK: blueprint {} → {}".format(
                    blueprint_args["blueprint_id"],
                    " → ".join(modules),
                )
            else:
                failed_module = evidence.get("failed_module") or "outcome feedback"
                response_text = "Failed: blueprint {} at {}".format(
                    blueprint_args["blueprint_id"], failed_module,
                )

            tool_call = {
                "function": "use_blueprint",
                "arguments": blueprint_args,
                "ok": bool(result.get("ok")),
                "blueprint_id": blueprint_args["blueprint_id"],
                "execution_id": result.get("execution_id", ""),
                "outcome_reported": bool(result.get("outcome_reported")),
                "evidence": evidence,
                "executions": executions,
                "result_preview": json.dumps(
                    {
                        "ok": result.get("ok"),
                        "closed_loop_ok": closed_loop_ok,
                        "evidence": evidence,
                    },
                    ensure_ascii=False,
                    default=str,
                )[:500],
            }
            cost_summary = self._cost_tracker.summary() if self._cost_tracker else None
            return ChatResponse(
                ok=closed_loop_ok,
                message=response_text,
                session_id=self._session_id,
                tool_calls=[tool_call],
                execution_results=executions,
                provider=self._config.provider,
                model="deterministic",
                rounds_used=0,
                cost=cost_summary,
            )

        # 2. Plan execution
        has_browser = bool(get_browser_status())
        steps = plan_execution(intent, has_browser=has_browser)
        if not steps:
            return None

        # 3. Execute through the same policy boundary as model-selected tools.
        results, summary = await execute_plan(steps, dispatch_fn)

        # 4. Check if execution succeeded
        ok = any(r["ok"] for r in results)
        # Build tool_calls list for audit
        tool_calls = [
            {"function": "execute_module", "module_id": r["module_id"],
             "ok": r["ok"], "error": r.get("error", ""),
             "model_route": (
                 deterministic_route.to_dict()
                 if deterministic_route is not None
                 else None
             )}
            for r in results
        ]

        # 5. Build response (deterministic template — zero LLM, zero hardcoded language)
        ok_modules = [r["module_id"] for r in results if r["ok"]]
        fail_modules = [(r["module_id"], r.get("error", "")) for r in results if not r["ok"]]

        if ok and not fail_modules:
            response_text = "OK: {}".format(" → ".join(ok_modules))
        elif ok:
            response_text = "OK: {}. Failed: {}".format(
                " → ".join(ok_modules),
                ", ".join("{} ({})".format(m, e[:50]) for m, e in fail_modules),
            )
        else:
            response_text = "Failed: {}".format(
                "; ".join("{}: {}".format(m, e[:80]) for m, e in fail_modules) or summary,
            )

        cost_summary = self._cost_tracker.summary() if self._cost_tracker else None

        return ChatResponse(
            ok=ok,
            message=response_text,
            session_id=self._session_id,
            tool_calls=tool_calls,
            execution_results=[r for r in results if r.get("module_id", "").startswith("browser.") or r.get("ok")],
            provider=self._config.provider,
            model="deterministic",
            rounds_used=1 if ok else 0,
            cost=cost_summary,
        )

    # ── Dispatch ──────────────────────────────────────────────────

    def _make_safe_dispatch(
        self,
        user_message: str = "",
        execute_blueprints: bool = True,
        blueprint_selection_mode: str = "model_selected",
        routing_decision: Optional[ToolIntentDecision] = None,
    ):
        """Create a dispatch function with permission + hooks + policy enforcement + assistant middleware."""
        self._ensure_routing_state()
        base_dispatch = self._dispatch_fn
        policies = self._policies
        enable_injection = self._config.enable_injection_detection
        enforcer = self._permission_enforcer
        hooks = self._hooks
        route_mode = routing_decision.mode if routing_decision else "action"

        # Wrap with assistant middleware (blueprint guard + selector healing)
        if self._assistant and base_dispatch:
            assisted_dispatch = self._assistant.wrap(base_dispatch, user_message)
        else:
            assisted_dispatch = base_dispatch

        async def record_block(
            func_name: str,
            func_args: dict,
            result: dict,
            policy_code: str,
        ) -> dict:
            """Forward outer policy denials to an evidence-aware executor."""
            recorder = getattr(
                getattr(self, "_tool_executor", None),
                "record_policy_denial",
                None,
            )
            if not callable(recorder):
                return result
            try:
                recorded = await recorder(
                    func_name,
                    func_args,
                    result,
                    policy_code=policy_code,
                )
            except Exception:
                recorded = None
            if isinstance(recorded, dict):
                return recorded
            failed = dict(result)
            failed["trace_error"] = "outer policy evidence could not be recorded"
            return failed

        async def preflight_blueprint_step(func_args: dict) -> dict:
            """Check static module access before a blueprint starts side effects."""
            decision = enforcer.check_route(
                "execute_module", func_args, route_mode,
            )
            if not decision.allowed:
                return await record_block("execute_module", func_args, {
                    "ok": False,
                    "error": decision.reason,
                    "policy_outcome": decision.outcome.value,
                    "routing_mode": route_mode,
                }, "blueprint_preflight_permission")
            if policies and not is_tool_allowed("execute_module", policies):
                return await record_block(
                    "execute_module",
                    func_args,
                    {"ok": False, "error": "Tool not allowed: execute_module"},
                    "blueprint_preflight_tool_policy",
                )
            if policies:
                module_id = func_args.get("module_id", "")
                if not is_module_allowed(module_id, policies):
                    category = (
                        module_id.split(".")[0] if "." in module_id else module_id
                    )
                    return await record_block("execute_module", func_args, {
                        "ok": False,
                        "error": "Module category '{}' is not allowed.".format(category),
                    }, "blueprint_preflight_module_policy")
            return {
                "ok": True,
                # Extension hooks remain runtime checks because their decisions
                # may depend on resolved params or external state.
                "dynamic_hooks_deferred": bool(hooks),
            }

        async def safe_dispatch(func_name: str, func_args: dict) -> dict:
            self._routing_metrics["tool_calls_attempted"] += 1

            # Conversation route + permission tier enforcement.  The exact
            # call is checked at runtime; MCP annotations are never authority.
            decision = enforcer.check_route(func_name, func_args, route_mode)
            if not decision.allowed:
                self._routing_metrics["tool_calls_blocked"] += 1
                return await record_block(func_name, func_args, {
                    "ok": False,
                    "error": decision.reason,
                    "policy_outcome": decision.outcome.value,
                    "routing_mode": route_mode,
                }, "agent_permission")

            if func_name == "use_blueprint":
                blueprint_id = str(func_args.get("blueprint_id", ""))
                resolver = getattr(self, "_trusted_blueprint_resolver", None)
                if resolver is None:
                    from flyto_ai.intelligence.planner import (
                        trusted_blueprint_summary,
                    )
                    trusted = trusted_blueprint_summary(blueprint_id)
                else:
                    trusted = resolver(blueprint_id)
                if not trusted:
                    self._routing_metrics["tool_calls_blocked"] += 1
                    return await record_block(func_name, func_args, {
                        "ok": False,
                        "error": (
                            "Blueprint blocked: automatic execution requires "
                            "verified runtime evidence."
                        ),
                        "policy_outcome": "block",
                        "routing_mode": route_mode,
                    }, "blueprint_trust")

            # Legacy policy enforcement (allowlists)
            if policies and not is_tool_allowed(func_name, policies):
                self._routing_metrics["tool_calls_blocked"] += 1
                return await record_block(
                    func_name,
                    func_args,
                    {"ok": False, "error": "Tool not allowed: {}".format(func_name)},
                    "agent_tool_policy",
                )
            if policies and func_name == "execute_module":
                module_id = func_args.get("module_id", "")
                if not is_module_allowed(module_id, policies):
                    category = module_id.split(".")[0] if "." in module_id else module_id
                    self._routing_metrics["tool_calls_blocked"] += 1
                    return await record_block(
                        func_name,
                        func_args,
                        {"ok": False, "error": "Module category '{}' is not allowed.".format(category)},
                        "agent_module_policy",
                    )

            # Extension hooks: before_tool_call (deny = short-circuit)
            if hooks:
                hook_result = await hooks.invoke_before_tool_call(func_name, func_args)
                if not hook_result.allowed:
                    self._routing_metrics["tool_calls_blocked"] += 1
                    return await record_block(
                        func_name,
                        func_args,
                        {"ok": False, "error": hook_result.reason},
                        "extension_hook",
                    )
                if hook_result.modified_arguments is not None:
                    func_args = hook_result.modified_arguments

            result = await assisted_dispatch(func_name, func_args)
            self._routing_metrics["tool_calls_executed"] += 1

            # An expanded blueprint is a deterministic execution contract.
            # Run it through this same safe dispatcher so every nested module
            # keeps permission, policy, hook, validation, and middleware checks.
            if (
                execute_blueprints
                and func_name == "use_blueprint"
                and isinstance(result, dict)
                and result.get("ok")
                and result.get("steps")
            ):
                from flyto_ai.blueprint_loop import execute_blueprint_loop

                result = await execute_blueprint_loop(
                    blueprint_id=func_args.get(
                        "blueprint_id", result.get("blueprint_id", ""),
                    ),
                    steps=result["steps"],
                    dispatch=safe_dispatch,
                    preflight=preflight_blueprint_step,
                    checkpoint_store=getattr(
                        self,
                        "_checkpoint_store",
                        None,
                    ),
                    max_repairs=getattr(
                        self._config,
                        "max_repair_attempts",
                        1,
                    ),
                    selection_mode=blueprint_selection_mode,
                )

            # Extension hooks: after_tool_call
            if hooks:
                await hooks.invoke_after_tool_call(func_name, func_args, result)

            # Pro: track tool call for budget enforcement
            if self._pro:
                try:
                    self._pro.record_tool_call()
                except Exception:
                    pass

            # Injection scanning
            if enable_injection and isinstance(result, dict):
                try:
                    from flyto_ai.prompt.injection_detector import scan_tool_result, format_warning_for_llm
                    import json as _json
                    result_text = _json.dumps(result, ensure_ascii=False, default=str)
                    if len(result_text) > 100:
                        warnings = scan_tool_result(func_name, result_text)
                        note = format_warning_for_llm(warnings)
                        if note:
                            result["_injection_warning"] = note
                except Exception:
                    pass

            return result

        return safe_dispatch if base_dispatch else None

    # ── Chat ──────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        template_context: Optional[Dict[str, Any]] = None,
        mode: str = "execute",
        on_tool_call=None,
        on_stream: Optional[StreamCallback] = None,
        dispatch_wrapper=None,
    ) -> ChatResponse:
        """Run one chat turn: message → tool loop → validation → response."""
        if self._closed:
            raise RuntimeError("agent is closed")
        t0 = time.monotonic()

        if not self._config.api_key and self._config.provider != "ollama":
            return ChatResponse(
                ok=False, message="No API key configured.",
                session_id=self._session_id, error="no_api_key",
            )

        # Injection detection
        injection_note = self._detect_injection(message)

        if self._transcript:
            self._transcript.record_user(message)

        await self._init_memory()

        routing_decision = (
            route_with_confirmation(message, history)
            if mode == "execute"
            else ToolIntentDecision(
                "action", 1.0, "explicit_non_execute_mode", (mode,),
            )
        )
        self._record_routing_decision(routing_decision)

        # ── Deterministic pipeline (try before LLM) ──
        if (
            mode == "execute"
            and routing_decision.tool_eligible
            and self._config.enable_deterministic
            and self._dispatch_fn
        ):
            det_result = await self._try_deterministic(
                message, on_tool_call, on_stream, dispatch_wrapper,
                routing_decision=routing_decision,
            )
            if det_result is not None:
                # Record cost, memory, audit
                duration_ms = int((time.monotonic() - t0) * 1000)
                await self._record_memory(message, det_result.message)
                self._emit_audit(
                    message, mode, det_result.tool_calls, det_result.execution_results,
                    det_result.ok, None, duration_ms, {},
                )
                return det_result

        model_route = self._select_model_route(message)

        messages = list(history or [])
        messages.append({"role": "user", "content": message})

        # Build dispatch + system prompt
        active_tools = self._tools_for_route(routing_decision, mode)
        dispatch_fn, has_tools = self._build_dispatch(
            message, on_tool_call, on_stream, dispatch_wrapper, mode=mode,
            routing_decision=routing_decision,
            active_tools=active_tools,
        )
        system_prompt, has_blueprint_match = await self._build_prompt(
            message, mode, has_tools, template_context, injection_note,
            history=history,
            routing_decision=routing_decision,
        )

        if self._compactor:
            messages, was_compacted = self._compactor.maybe_compact(messages)
            if was_compacted:
                logger.info("Context compacted before LLM call")

        # Call LLM
        total_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
        total_rounds = 0

        if has_tools:
            response_content, tool_calls, rounds_used, usage_dict = await self._call_llm(
                messages, system_prompt, dispatch_fn, on_stream=on_stream,
                tools=active_tools,
            )
        else:
            response_content, tool_calls, rounds_used, usage_dict = await self._call_llm_toolless(
                messages, system_prompt, on_stream=on_stream,
            )
        total_rounds += rounds_used
        for k in total_usage:
            total_usage[k] += usage_dict.get(k, 0)

        # Cost tracking
        self._record_cost(usage_dict)

        if not response_content:
            duration_ms = int((time.monotonic() - t0) * 1000)
            self._emit_audit(message, mode, tool_calls, [], False, "provider_call_failed", duration_ms, total_usage)
            if self._transcript:
                self._transcript.record_error("provider_call_failed")
            return ChatResponse(
                ok=False, message="AI provider call failed. Please try again.",
                session_id=self._session_id, error="provider_call_failed",
            )

        # System safety net: if LLM only used discovery tools (search/list)
        # but never executed anything, nudge once. Only accept if retry
        # results in actual execution.
        # A Space's own workflows arrive under their own names (kintone), so
        # "did the model do anything" has to count them; otherwise a nudge
        # that made the model run one would be thrown away after it ran.
        runnable_names, workflow_names = self._runnable_tool_names(active_tools)
        _action_tools = {
            "execute_module", "use_blueprint", "navigate_website", "ask_user",
        } | set(workflow_names)
        _has_action = any(tc.get("function") in _action_tools for tc in tool_calls)
        if (mode == "execute"
                and self._assistant
                and not _has_action
                and response_content
                and has_tools
                and routing_decision.tool_eligible
                and total_rounds <= 1
                and self._config.provider != "ollama"):
            try:
                nudge_messages = messages + [
                    {"role": "assistant", "content": response_content},
                    {"role": "user", "content": (
                        "If this task requires you to actually DO something (go to a website, "
                        "execute a module, automate an action), use the tools available. "
                        "If this is just a knowledge question, answer as you did."
                    )},
                ]
                retry_content, retry_tc, retry_rounds, retry_usage = await self._call_llm(
                    nudge_messages, system_prompt, dispatch_fn, on_stream=on_stream,
                    tools=active_tools,
                )
                # Only accept if LLM actually EXECUTED something (not just searched)
                has_execution = any(
                    tc.get("function") in (
                        {"execute_module", "use_blueprint", "ask_user"}
                        | set(workflow_names)
                    )
                    for tc in retry_tc
                )
                if has_execution:
                    logger.info("Nudge accepted: LLM used execution tools")
                    response_content = retry_content
                    tool_calls = retry_tc
                    total_rounds += retry_rounds
                    for k in total_usage:
                        total_usage[k] += retry_usage.get(k, 0)
            except Exception:
                pass

        # The nudge above is a request the model may ignore; this is the rule.
        if mode == "execute" and has_tools and routing_decision.tool_eligible:
            response_content, tool_calls, total_rounds, total_usage = (
                await self._guard_narrated_execution(
                    message, response_content, tool_calls, messages,
                    system_prompt, dispatch_fn, on_stream, active_tools,
                    runnable_names, workflow_names, total_rounds, total_usage,
                )
            )

        # YAML mode: nudge + validation
        if mode == "yaml":
            response_content, tool_calls, total_rounds, total_usage = await self._handle_yaml_validation(
                response_content, tool_calls, messages, system_prompt, dispatch_fn,
                total_rounds, total_usage,
            )

        # Collect execution results
        execution_results = []
        for tool_call in tool_calls:
            if tool_call.get("function") == "execute_module":
                execution_results.append(tool_call)
            elif tool_call.get("function") == "use_blueprint":
                nested = tool_call.get("executions", [])
                if isinstance(nested, list):
                    execution_results.extend(
                        item for item in nested if isinstance(item, dict)
                    )

        # Guard: if ALL executions failed, force LLM to acknowledge
        response_content, total_rounds, total_usage = await self._handle_failure_guard(
            response_content, execution_results, messages, system_prompt,
            on_stream, total_rounds, total_usage,
        )

        # Assistant post-process: blueprint feedback + output auto-save + pending input
        pending_input = None
        if self._assistant:
            pending_input = await self._assistant.post_process(
                tool_calls, execution_results, message, mode,
                dispatch=self._dispatch_fn,
            )

        # Memory + transcript recording
        await self._record_memory(message, response_content)
        self._record_transcript(response_content, tool_calls)

        usage = UsageStats(**total_usage) if any(v > 0 for v in total_usage.values()) else None
        duration_ms = int((time.monotonic() - t0) * 1000)
        self._emit_audit(message, mode, tool_calls, execution_results, True, None, duration_ms, total_usage)
        cost_summary = self._cost_tracker.summary() if self._cost_tracker else None

        return ChatResponse(
            ok=True, message=response_content, session_id=self._session_id,
            tool_calls=tool_calls, execution_results=execution_results,
            provider=model_route.provider or self._config.provider,
            model=model_route.model or self._config.resolved_model,
            rounds_used=total_rounds, usage=usage, cost=cost_summary,
            pending_input=pending_input,
        )

    # ── Chat phase helpers ────────────────────────────────────────

    def _runnable_tool_names(
        self, active_tools: Optional[List[Dict]],
    ) -> Tuple[set, List[str]]:
        """Names this turn could actually run, and which of them are the
        Space's own workflows (registered by the caller, unknown to the static
        permission map). Read-only tools only look things up."""
        enforcer = self._permission_enforcer
        runnable: set = set()
        workflows: List[str] = []
        for tool in active_tools or []:
            name = self._tool_name(tool)
            if not name:
                continue
            if enforcer.required_level(name, {}) <= PermissionLevel.READ_ONLY:
                continue
            runnable.add(name)
            if name not in TOOL_PERMISSION_MAP:
                workflows.append(name)
        return runnable, workflows

    async def _guard_narrated_execution(
        self, message, response_content, tool_calls, messages, system_prompt,
        dispatch_fn, on_stream, active_tools, runnable_names, workflow_names,
        total_rounds, total_usage,
    ):
        """Never let a reply present a run that this turn did not call.

        The user said 幫我登入kintone. The model answered 「我將執行 "kintone"
        工作流程來幫助您登入。請稍候。執行中...」 and called nothing; the audit
        for that turn shows tool_calls_count=0 and the workflow only ran after
        the user typed 執行啊. tool_choice was "auto" because the provider's
        browser-task heuristic does not know a Space's workflow names, and the
        discovery nudge is a polite request the model is free to ignore. So:
        when the turn has something it can run and the reply reads as a
        commitment without a call, retry once with the call forced; if the
        model still narrates -- or the provider cannot force -- replace the
        reply with a plain statement that nothing ran and what could be run.
        """
        if not runnable_names or not response_content:
            return response_content, tool_calls, total_rounds, total_usage
        if any(tc.get("function") in runnable_names for tc in tool_calls):
            return response_content, tool_calls, total_rounds, total_usage
        if not _reads_as_commitment(response_content):
            return response_content, tool_calls, total_rounds, total_usage

        if getattr(self._provider, "supports_forced_tool_choice", False):
            retry_content, retry_tc, retry_rounds, retry_usage = await self._call_llm(
                messages, system_prompt, dispatch_fn, on_stream=on_stream,
                tools=active_tools, tool_choice="required",
            )
            total_rounds += retry_rounds
            for k in total_usage:
                total_usage[k] += retry_usage.get(k, 0)
            if any(tc.get("function") in runnable_names for tc in retry_tc):
                logger.info(
                    "Forced tool choice accepted: model called %s after narrating",
                    [tc.get("function") for tc in retry_tc],
                )
                return retry_content or "", retry_tc, total_rounds, total_usage
            if retry_tc:
                # Lookups the forced round did make really happened; keep
                # them in the log even though nothing ran.
                tool_calls = retry_tc

        self._ensure_routing_state()
        language = detect_language(message, preferred_language=self._preferred_language)
        logger.warning(
            "Execute-mode reply narrated a run without calling a tool; "
            "replaced with a statement that nothing ran (runnable: %s)",
            sorted(runnable_names),
        )
        return (
            _nothing_ran_message(language, workflow_names),
            tool_calls, total_rounds, total_usage,
        )

    def _ensure_routing_state(self) -> None:
        """Support lightweight test agents created without ``__init__``."""
        if not hasattr(self, "_preferred_language"):
            self._preferred_language = None
        if not hasattr(self, "_last_routing_decision"):
            self._last_routing_decision = None
        if not hasattr(self, "_routing_metrics"):
            self._routing_metrics = {
                "turns": 0,
                "answer_only_turns": 0,
                "ambiguous_turns": 0,
                "action_turns": 0,
                "tool_calls_attempted": 0,
                "tool_calls_executed": 0,
                "tool_calls_blocked": 0,
            }

    def _record_routing_decision(
        self,
        decision: ToolIntentDecision,
    ) -> None:
        self._ensure_routing_state()
        self._last_routing_decision = decision
        self._routing_metrics["turns"] += 1
        key = "{}_turns".format(decision.mode)
        if key in self._routing_metrics:
            self._routing_metrics[key] += 1

    @staticmethod
    def _tool_name(tool: Dict[str, Any]) -> str:
        """Read either MCP-style or OpenAI-style tool definitions."""
        if tool.get("name"):
            return str(tool["name"])
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("name", ""))
        return ""

    def _tools_for_route(
        self,
        decision: ToolIntentDecision,
        mode: str,
    ) -> List[Dict]:
        """Expose the smallest schema set justified by this turn."""
        tools = list(self._tools or [])
        if mode != "execute":
            return tools
        if decision.mode == "answer_only":
            return []

        enforcer = self._permission_enforcer
        maximum = (
            PermissionLevel.READ_ONLY
            if decision.mode == "ambiguous"
            else enforcer.level
        )
        return [
            tool for tool in tools
            if enforcer.required_level(self._tool_name(tool), {}) <= maximum
        ]

    def _resolve_reply_language(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self._ensure_routing_state()
        preferred = self._preferred_language
        if preferred is None:
            for item in reversed(history or []):
                if item.get("role") != "user":
                    continue
                content = str(item.get("content", "")).strip()
                if len(content) >= 15 or any("\u4e00" <= c <= "\u9fff" for c in content):
                    preferred = detect_language(content)
                    break
        language = detect_language(message, preferred_language=preferred)
        meaningful_length = sum(char.isalnum() for char in message)
        if (
            preferred is not None
            and meaningful_length <= 4
            and language != preferred
        ):
            language = preferred
        self._preferred_language = language
        return language

    def _detect_injection(self, message: str) -> Optional[str]:
        """Scan user message for prompt injection patterns."""
        if not self._config.enable_injection_detection:
            return None
        try:
            from flyto_ai.prompt.injection_detector import scan_text, format_warning_for_llm
            warnings = scan_text(message, source="user_input")
            return format_warning_for_llm(warnings)
        except Exception:
            return None

    def _build_dispatch(
        self, message: str, on_tool_call, on_stream, dispatch_wrapper,
        mode: str = "execute",
        blueprint_selection_mode: str = "model_selected",
        routing_decision: Optional[ToolIntentDecision] = None,
        active_tools: Optional[List[Dict]] = None,
    ) -> Tuple:
        """Build the final dispatch function with middleware + instrumentation.

        Returns:
            (dispatch_fn, has_tools)
        """
        dispatch_fn = self._make_safe_dispatch(
            user_message=message,
            execute_blueprints=mode == "execute",
            blueprint_selection_mode=blueprint_selection_mode,
            routing_decision=routing_decision,
        )
        if dispatch_wrapper and dispatch_fn:
            dispatch_fn = dispatch_wrapper(dispatch_fn)
        if dispatch_fn and (on_tool_call or on_stream):
            _base = dispatch_fn

            async def _instrumented(func_name: str, func_args: dict) -> dict:
                if on_tool_call:
                    try:
                        on_tool_call(func_name, func_args)
                    except Exception:
                        pass
                if on_stream:
                    try:
                        on_stream(StreamEvent(type=StreamEventType.TOOL_START, tool_name=func_name, tool_args=func_args))
                    except Exception:
                        pass
                result = await _base(func_name, func_args)
                if on_stream:
                    try:
                        on_stream(StreamEvent(type=StreamEventType.TOOL_END, tool_name=func_name, tool_result=result if isinstance(result, dict) else None))
                    except Exception:
                        pass
                return result

            dispatch_fn = _instrumented
        visible_tools = self._tools if active_tools is None else active_tools
        has_tools = bool(visible_tools and dispatch_fn)
        return dispatch_fn, has_tools

    async def _build_prompt(
        self, message: str, mode: str, has_tools: bool,
        template_context: Optional[Dict[str, Any]],
        injection_note: Optional[str],
        history: Optional[List[Dict[str, Any]]] = None,
        routing_decision: Optional[ToolIntentDecision] = None,
    ) -> Tuple[str, bool]:
        """Build the system prompt with memory, injection notes, and blueprint hints.

        Returns:
            (system_prompt, has_blueprint_match)
        """
        if self._system_prompt:
            return self._system_prompt, False

        reply_language = self._resolve_reply_language(message, history)

        memory_addition = None
        if self._memory_search:
            try:
                relevant = await self._memory_search.search(message, top_k=3)
                if relevant:
                    memory_lines = ["## Relevant Memory (from past conversations):"]
                    for r in relevant:
                        memory_lines.append("- {}".format(r["content"][:300]))
                    memory_addition = "\n".join(memory_lines)
            except Exception as e:
                logger.debug("Memory search failed: %s", e)

        blueprint_hint = ""
        if (
            self._assistant
            and mode == "execute"
            and has_tools
            and (routing_decision is None or routing_decision.tool_eligible)
        ):
            blueprint_hint = self._assistant.prepare(message, mode)
            self._last_blueprint_hint = blueprint_hint

        combined_addition = memory_addition or ""
        if injection_note:
            combined_addition = (combined_addition + "\n\n" + injection_note).strip()
        if blueprint_hint:
            combined_addition = (combined_addition + "\n\n" + blueprint_hint).strip()

        # Browser session hint — tell LLM if browser is already running
        try:
            from flyto_ai.tools.core_tools import get_browser_status
            browser_hint = get_browser_status()
            if browser_hint:
                combined_addition = (combined_addition + "\n\n" + browser_hint).strip()
        except Exception:
            pass

        from flyto_ai.package_metadata import runtime_module_count

        _module_count = runtime_module_count() or 0

        # Pro: inject catalog outline for LLM module discovery
        if self._pro and self._config.enable_knowledge:
            try:
                catalog = self._pro.get_catalog_outline()
                if catalog:
                    catalog_section = "\n\n## Module Catalog:\n" + catalog
                    combined_addition = (combined_addition + catalog_section).strip()
            except Exception:
                pass

        prompt = build_system_prompt(
            module_count=_module_count, context=template_context, has_tools=has_tools,
            mode=mode, reply_language=reply_language,
            admin_addition=combined_addition or None,
        )
        return prompt, bool(blueprint_hint)

    async def _handle_yaml_validation(
        self, response_content: str, tool_calls: List[Dict],
        messages: List[Dict], system_prompt: str, dispatch_fn,
        total_rounds: int, total_usage: Dict[str, int],
    ) -> Tuple[str, List[Dict], int, Dict[str, int]]:
        """Nudge LLM to produce YAML and validate it iteratively.

        Returns:
            (response_content, tool_calls, total_rounds, total_usage)
        """
        if not extract_yaml_from_response(response_content):
            nudge_messages = messages + [
                {"role": "assistant", "content": response_content},
                {"role": "user", "content": "You must always output a Flyto2 Workflow YAML. Please generate the workflow YAML now using the modules and blueprints available."},
            ]
            nudge_content, nudge_tc, nudge_rounds, nudge_usage = await self._call_llm(nudge_messages, system_prompt, dispatch_fn)
            total_rounds += nudge_rounds
            for k in total_usage:
                total_usage[k] += nudge_usage.get(k, 0)
            if nudge_content and extract_yaml_from_response(nudge_content):
                response_content = nudge_content
                tool_calls.extend(nudge_tc)

        for _attempt in range(self._config.max_validation_rounds):
            yaml_str = extract_yaml_from_response(response_content)
            if not yaml_str:
                break

            # Use deep validation when pro bridge is available
            if self._pro and self._config.enable_contract_validation:
                try:
                    from flyto_ai.validation import validate_workflow_deep
                    deep_result = await validate_workflow_deep(yaml_str, self._pro)
                    errors = deep_result["basic"] + deep_result["contract"]

                    # Auto-generate missing modules via EvolutionRouter
                    if self._config.enable_evolution and deep_result["missing_modules"]:
                        gen_result = await self._pro.generate_missing_modules(
                            deep_result["missing_modules"],
                            context=messages[-1].get("content", "") if messages else "",
                        )
                        if gen_result and gen_result.get("all_generated"):
                            logger.info("Auto-generated %d missing modules",
                                        len(gen_result.get("generated", [])))
                            errors = validate_workflow_steps(yaml_str)
                except Exception as e:
                    logger.debug("Deep validation failed, falling back to basic: %s", e)
                    errors = validate_workflow_steps(yaml_str)
            else:
                errors = validate_workflow_steps(yaml_str)

            if not errors:
                break
            error_list = "\n".join("- {}".format(e) for e in errors)
            retry_messages = messages + [
                {"role": "assistant", "content": response_content},
                {"role": "user", "content": "The workflow YAML you generated has validation errors:\n{}\n\nPlease call get_module_info() for each failing module to verify the correct param names, then regenerate the YAML.".format(error_list)},
            ]
            retry_content, retry_tc, retry_rounds, retry_usage = await self._call_llm(retry_messages, system_prompt, dispatch_fn)
            total_rounds += retry_rounds
            for k in total_usage:
                total_usage[k] += retry_usage.get(k, 0)
            if retry_content:
                response_content = retry_content
                tool_calls.extend(retry_tc)
            else:
                break

        return response_content, tool_calls, total_rounds, total_usage

    async def _handle_failure_guard(
        self, response_content: str, execution_results: List[Dict],
        messages: List[Dict], system_prompt: str,
        on_stream: Optional[StreamCallback],
        total_rounds: int, total_usage: Dict[str, int],
    ) -> Tuple[str, int, Dict[str, int]]:
        """If ALL executions failed, force LLM to acknowledge failures.

        Returns:
            (response_content, total_rounds, total_usage)
        """
        if not execution_results or not all(not er.get("ok", False) for er in execution_results):
            return response_content, total_rounds, total_usage

        # NOTE: Individual execution errors are already recorded by
        # AssistantMiddleware._on_result() → EMS. No duplicate recording here.

        errors = []
        for er in execution_results:
            preview = er.get("result_preview", "")
            try:
                err_data = json.loads(preview) if preview.startswith("{") else {}
                err_msg = err_data.get("error", "")
            except Exception:
                err_msg = ""
            errors.append("{}: {}".format(er.get("module_id", "?"), err_msg or "failed"))
        error_detail = "\n".join(errors)
        correction_messages = messages + [
            {"role": "assistant", "content": response_content},
            {"role": "user", "content": (
                "STOP. All {} module executions FAILED:\n{}\n\n"
                "Your previous response is WRONG — you must NOT claim success. "
                "Rewrite your response: (1) state which modules failed and why, "
                "(2) suggest what the user can do to fix it. "
                "Do NOT fabricate results or URLs."
            ).format(len(execution_results), error_detail)},
        ]
        corrected, _, corr_rounds, corr_usage = await self._call_llm_toolless(
            correction_messages, system_prompt, on_stream=on_stream,
        )
        if corrected:
            response_content = corrected
            total_rounds += corr_rounds
            for k in total_usage:
                total_usage[k] += corr_usage.get(k, 0)

        return response_content, total_rounds, total_usage

    async def _record_memory(self, message: str, response_content: str) -> None:
        """Record conversation turn in memory store, summarizer, and search index."""
        session_id = self._session_id
        if self._memory_store:
            try:
                await self._memory_store.add_message(session_id, "user", message)
                await self._memory_store.add_message(session_id, "assistant", response_content)
            except Exception as e:
                logger.debug("Memory store failed: %s", e)
        if self._summarizer and self._memory_store:
            try:
                await self._summarizer.maybe_summarize(session_id, self._memory_store)
            except Exception as e:
                logger.debug("Summarization failed: %s", e)
        if self._memory_search:
            try:
                exchange = "User: {}\nAssistant: {}".format(message[:200], response_content[:200])
                await self._memory_search.index_content(session_id, exchange)
            except Exception as e:
                logger.debug("Memory indexing failed: %s", e)

    def _record_transcript(self, response_content: str, tool_calls: List[Dict]) -> None:
        """Record assistant response and tool calls to transcript."""
        if not self._transcript:
            return
        self._transcript.record_assistant(response_content, provider=self._config.provider, model=self._config.resolved_model)
        for tc in tool_calls:
            self._transcript.record_tool_call(tc.get("function", ""), tc.get("arguments", {}))
            if tc.get("function") == "execute_module":
                self._transcript.record_execution(tc.get("module_id", ""), tc.get("ok", False), tc.get("result_preview", ""))
        if self._cost_tracker:
            self._transcript.record_meta({"event": "cost_summary", **self._cost_tracker.summary()})

    def _record_cost(self, usage_dict: Dict[str, int]) -> None:
        """Record token usage in cost tracker + pro CostController."""
        if not any(v > 0 for v in usage_dict.values()):
            return
        prompt_tokens = usage_dict.get("prompt_tokens", 0)
        completion_tokens = usage_dict.get("completion_tokens", 0)

        if self._cost_tracker:
            try:
                self._cost_tracker.record(
                    model=self._config.resolved_model,
                    provider=self._config.provider or "openai",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_read_tokens=usage_dict.get("cache_read_input_tokens", 0),
                )
            except Exception as e:
                logger.debug("Cost tracking failed: %s", e)

        # Pro CostController — multi-resource budget enforcement
        if self._pro:
            try:
                self._pro.record_llm_usage(
                    self._config.resolved_model, prompt_tokens, completion_tokens,
                )
            except Exception as e:
                if "exceeded" in type(e).__name__.lower():
                    logger.warning("Pro budget exceeded: %s", e)
                    raise
                logger.debug("Pro cost tracking failed: %s", e)

    # ── Internal ──────────────────────────────────────────────────

    def _emit_audit(self, user_message, mode, tool_calls, execution_results, ok, error, duration_ms, usage):
        try:
            from flyto_ai.audit import ChatAuditEntry
            cost_usd = 0.0
            if self._cost_tracker:
                cost_usd = self._cost_tracker.session_total_usd
            routed_model = (
                self._last_model_route.model
                if self._last_model_route is not None
                else self._config.resolved_model
            )
            ChatAuditEntry(
                user_message=user_message[:200], provider=self._config.provider or "openai",
                model=routed_model, mode=mode,
                tool_calls_count=len(tool_calls), execution_count=len(execution_results),
                duration_ms=duration_ms, prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0), ok=ok, error=error,
                tool_calls=tool_calls, execution_results=execution_results,
                cost_usd=cost_usd,
            ).emit()
        except Exception:
            pass

    async def _call_llm(
        self,
        messages,
        system_prompt,
        dispatch_fn,
        on_stream=None,
        tools=None,
        tool_choice=None,
    ):
        try:
            active_tools = self._tools if tools is None else tools
            kwargs = {"on_stream": on_stream}
            # Passed only when set, so a client written before the argument
            # existed keeps working on every ordinary call.
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            return await self._provider.chat(
                messages, system_prompt, active_tools,
                dispatch_fn, self._config.max_tool_rounds,
                **kwargs,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None, [], 0, {}

    async def _call_llm_toolless(self, messages, system_prompt, on_stream=None):
        try:
            async def _noop_dispatch(name, args):
                return {"ok": False, "error": "No tools available"}
            return await self._provider.chat(
                messages, system_prompt, [], _noop_dispatch, max_rounds=1,
                on_stream=on_stream,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None, [], 0, {}
