# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""ProBridge — dual-mode bridge to flyto-pro intelligence.

Two layers:
1. flyto-pro-core (open source, always available):
   - ContractEngine  — deep workflow validation + binding resolution
   - CostController  — multi-resource budget management
   - Interfaces      — ILLMService, IVectorStoreRepository

2. flyto-pro (licensed, premium features):
   - EMSRouter       — error memory system (learn from failures)
   - KnowledgeRouter — module discovery by semantic search
   - EvolutionRouter — auto-generate missing modules

All modules are lazy-loaded. Open-source features work without a license.
Premium features require a valid license in ~/.flyto2/license/license.json.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cached singletons
_contract_engine = None
_contract_engine_checked = False
_cost_controller = None
_ems_router = None
_knowledge_router = None
_evolution_router = None


def _check_core_available() -> bool:
    """Check if flyto-pro-core (open source) is installed."""
    try:
        import flyto_pro_core  # noqa: F401
        return True
    except ImportError:
        return False


def _check_pro_available() -> bool:
    """Check if flyto-pro (licensed) is installed."""
    try:
        import pro  # noqa: F401
        return True
    except ImportError:
        return False


def _check_license_tier() -> str:
    """Get license tier. Returns 'free' if unlicensed."""
    try:
        from pro.license.checker import get_tier
        return get_tier()
    except Exception:
        return "free"


class ProBridge:
    """Unified bridge to flyto-pro intelligence modules.

    Lazy-inits each module on first access. Thread-safe via module-level
    singletons. Graceful degradation at every level:
    - No flyto-pro-core → all methods return None
    - flyto-pro-core only → contract + cost work, EMS/evolution return None
    - flyto-pro + free tier → same as core only
    - flyto-pro + pro/enterprise tier → all features enabled
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._core_available = _check_core_available()
        self._pro_available = _check_pro_available()
        self._license_tier = _check_license_tier() if self._pro_available else "free"

        if self._core_available:
            logger.info("flyto-pro-core enabled (open source)")
        if self._pro_available and self._license_tier != "free":
            logger.info("flyto-pro enabled (tier=%s)", self._license_tier)
        elif self._pro_available:
            logger.debug("flyto-pro installed but license=free — premium features disabled")

    @property
    def available(self) -> bool:
        """True if at least flyto-pro-core is available."""
        return self._core_available

    @property
    def premium_available(self) -> bool:
        """True if flyto-pro is installed AND licensed (pro/enterprise)."""
        return self._pro_available and self._license_tier in ("pro", "enterprise")

    @property
    def license_tier(self) -> str:
        return self._license_tier

    # ── Contract Engine (OPEN SOURCE — flyto-pro-core) ───────────

    def get_contract_engine(self):
        """Get or create the ContractEngine singleton."""
        global _contract_engine, _contract_engine_checked
        if _contract_engine_checked:
            return _contract_engine
        _contract_engine_checked = True

        if not self._core_available:
            return None
        try:
            from flyto_pro_core.contract.engine import ContractEngine
            _contract_engine = ContractEngine()
            logger.debug("ContractEngine initialized (open source)")
        except Exception as e:
            logger.debug("ContractEngine init failed: %s", e)
            _contract_engine = None
        return _contract_engine

    async def initialize_contract_engine(self) -> bool:
        """Initialize the contract engine (loads module catalog from flyto-core)."""
        engine = self.get_contract_engine()
        if engine is None:
            return False
        try:
            await engine.initialize()
            return True
        except Exception as e:
            logger.debug("ContractEngine.initialize() failed: %s", e)
            return False

    async def validate_workflow_deep(self, yaml_str: str) -> Optional[Dict[str, Any]]:
        """Validate a workflow YAML using ContractEngine."""
        engine = self.get_contract_engine()
        if engine is None:
            return None
        try:
            import yaml as yaml_lib
            from flyto_pro_core.contract.models.workflow_spec import WorkflowSpec

            workflow_data = yaml_lib.safe_load(yaml_str)
            if not isinstance(workflow_data, dict):
                return None

            spec = WorkflowSpec(**workflow_data)
            report = await engine.validate_workflow(spec)
            return {
                "valid": report.valid if hasattr(report, "valid") else not report.issues,
                "issues": [
                    {"severity": str(getattr(i, "severity", "error")),
                     "message": str(getattr(i, "message", str(i))),
                     "node_id": getattr(i, "node_id", None)}
                    for i in (report.issues if hasattr(report, "issues") else [])
                ],
            }
        except Exception as e:
            logger.debug("Deep validation failed: %s", e)
            return None

    # ── Cost Controller (OPEN SOURCE — flyto-pro-core) ───────────

    def get_cost_controller(self):
        """Get or create the CostController singleton."""
        global _cost_controller
        if _cost_controller is not None:
            return _cost_controller
        if not self._core_available:
            return None
        try:
            from flyto_pro_core.cost.controller import CostController, BudgetConfig

            cfg = self._config
            tier = getattr(cfg, "pro_budget_tier", "") if cfg else ""
            if tier:
                budget = BudgetConfig.for_tier(tier)
            else:
                budget = BudgetConfig.from_env()
            _cost_controller = CostController(budget=budget)
            logger.debug("CostController initialized (tier=%s)", tier or "env")
        except Exception as e:
            logger.debug("CostController init failed: %s", e)
            _cost_controller = None
        return _cost_controller

    def record_llm_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int,
    ) -> Optional[float]:
        """Record LLM usage in CostController."""
        controller = self.get_cost_controller()
        if controller is None:
            return None
        try:
            return controller.record_llm_usage(model, prompt_tokens, completion_tokens)
        except Exception as e:
            if "exceeded" in type(e).__name__.lower():
                raise
            logger.debug("record_llm_usage failed: %s", e)
            return None

    def record_tool_call(self) -> None:
        """Record a tool call in CostController."""
        controller = self.get_cost_controller()
        if controller is None:
            return
        try:
            controller.record_tool_call()
        except Exception:
            pass

    def check_budget(self) -> None:
        """Check budget — raises BudgetExceededError if over limit."""
        controller = self.get_cost_controller()
        if controller is None:
            return
        controller.check_budget()

    def get_cost_summary(self) -> Optional[Dict[str, Any]]:
        """Get cost summary."""
        controller = self.get_cost_controller()
        if controller is None:
            return None
        try:
            return controller.get_summary()
        except Exception:
            return None

    # ── Error Memory System (LICENSED — flyto-pro) ───────────────

    def get_ems(self):
        """Get or create the EMSRouter singleton. Requires pro/enterprise license."""
        global _ems_router
        if _ems_router is not None:
            return _ems_router
        if not self.premium_available:
            return None
        try:
            from pro.ems.router.router import EMSRouter
            _ems_router = EMSRouter(use_vector_db=True)
            logger.debug("EMSRouter initialized (licensed)")
        except Exception as e:
            logger.debug("EMSRouter init failed (Qdrant may be unavailable): %s", e)
            try:
                from pro.ems.router.router import EMSRouter
                _ems_router = EMSRouter(use_vector_db=False)
                logger.debug("EMSRouter initialized (in-memory fallback)")
            except Exception as e2:
                logger.debug("EMSRouter init fully failed: %s", e2)
                _ems_router = None
        return _ems_router

    async def record_error(
        self,
        error_type: str,
        message: str,
        stage: str,
        module_id: str = "",
        code_snippet: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Record an error in EMS for future learning."""
        ems = self.get_ems()
        if ems is None:
            return None
        try:
            return await ems.record_error(
                error_type=error_type, message=message,
                stage=stage, module_id=module_id,
                code_snippet=code_snippet,
            )
        except Exception as e:
            logger.debug("EMS record_error failed: %s", e)
            return None

    async def get_lesson_for_error(
        self, error_type: str, message: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up a known fix for an error."""
        ems = self.get_ems()
        if ems is None:
            return None
        try:
            return await ems.get_lesson_for_error(
                error_type=error_type, message=message,
            )
        except Exception as e:
            logger.debug("EMS lesson lookup failed: %s", e)
            return None

    # ── Knowledge Router (LICENSED — flyto-pro) ──────────────────

    def get_knowledge_router(self):
        """Get or create the KnowledgeRouter. Requires pro/enterprise license."""
        global _knowledge_router
        if _knowledge_router is not None:
            return _knowledge_router
        if not self.premium_available:
            return None
        try:
            from pro.knowledge.evolution_router.router import EvolutionRouter as KnowledgeRouter
            _knowledge_router = KnowledgeRouter()
            logger.debug("KnowledgeRouter initialized (licensed)")
        except Exception as e:
            logger.debug("KnowledgeRouter init failed: %s", e)
            _knowledge_router = None
        return _knowledge_router

    async def search_modules_smart(
        self, query: str, context: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Search for relevant modules using the knowledge layer."""
        kr = self.get_knowledge_router()
        if kr is not None:
            try:
                result = await kr.generate(missing_modules=[query], context=context)
                if hasattr(result, "generated_modules") and result.generated_modules:
                    return [m.module_id for m in result.generated_modules if hasattr(m, "module_id")]
            except Exception as e:
                logger.debug("KnowledgeRouter search failed: %s", e)

        # Fallback: ContractEngine catalog (open source)
        engine = self.get_contract_engine()
        if engine is not None:
            try:
                outline = engine.get_catalog_outline()
                if outline and hasattr(outline, "categories"):
                    q_lower = query.lower()
                    matches = []
                    for cat in outline.categories:
                        cat_name = getattr(cat, "name", str(cat))
                        if q_lower in cat_name.lower():
                            matches.append(cat_name)
                    if matches:
                        return matches[:5]
            except Exception:
                pass
        return None

    # ── Evolution Router (LICENSED — flyto-pro) ──────────────────

    def get_evolution_router(self):
        """Get or create the EvolutionRouter. Requires pro/enterprise license."""
        global _evolution_router
        if _evolution_router is not None:
            return _evolution_router
        if not self.premium_available:
            return None
        try:
            from pro.knowledge.evolution_router.router import EvolutionRouter
            ems = self.get_ems()
            _evolution_router = EvolutionRouter(ems=ems)
            logger.debug("EvolutionRouter initialized (licensed, ems=%s)", ems is not None)
        except Exception as e:
            logger.debug("EvolutionRouter init failed: %s", e)
            _evolution_router = None
        return _evolution_router

    async def generate_missing_modules(
        self, module_ids: List[str], context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate missing modules via EvolutionRouter."""
        router = self.get_evolution_router()
        if router is None:
            return None
        try:
            result = await router.generate(module_ids, context=context)
            return {
                "all_generated": result.all_generated,
                "generated": [
                    {"module_id": getattr(m, "module_id", ""), "quality_score": getattr(m, "quality_score", 0)}
                    for m in (result.generated_modules or [])
                ],
                "failed": result.failed_modules or [],
                "generation_time_ms": getattr(result, "generation_time_ms", 0),
            }
        except Exception as e:
            logger.debug("Module generation failed: %s", e)
            return None

    # ── Catalog (OPEN SOURCE — flyto-pro-core) ───────────────────

    def get_catalog_outline(self) -> Optional[str]:
        """Get a compact module catalog outline for the system prompt."""
        engine = self.get_contract_engine()
        if engine is None:
            return None
        try:
            outline = engine.get_catalog_outline()
            if not outline:
                return None
            lines = []
            for cat in getattr(outline, "categories", []):
                name = getattr(cat, "name", str(cat))
                count = getattr(cat, "module_count", 0)
                desc = getattr(cat, "description", "")
                lines.append("- **{}** ({}) — {}".format(name, count, desc))
            return "\n".join(lines) if lines else None
        except Exception:
            return None
