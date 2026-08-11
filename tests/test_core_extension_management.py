# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for the host-only flyto-core extension management adapter.

The file has two halves and the split is deliberate.

`TestRealCoreContract` binds the **installed** `core.plugin.loader` and asserts
every name this bridge depends on actually exists there: the four exports, the
loader methods, the `ExtensionResult` fields the envelope publishes, and the
shape of `EXTENSION_KINDS`. A fixture cannot catch a Core rename, because a
fixture agrees with whatever the host already believes — that is exactly how the
capability-manifest bridge once validated a shape Core never emitted and
reported a fully installed Core as empty. These tests fail loudly instead.

The behaviour half drives the host logic through a fake loader that answers in
the real contract's shape: `list_extensions` returns a plain list, `install` and
`uninstall` return a result object read structurally, and kinds come from a
records constant. Three invariants are load-bearing and each is tested on its
own rather than implied by a happy path:

* installation is host authority — never in a model-facing tool catalog and
  refused at the MCP dispatch boundary;
* mutation is opt-in — with `FLYTO_EXTENSIONS_INSTALL_ENABLED` unset, Core is
  never reached at all, for any input;
* the envelope is fixed and installer output never appears in it.
"""
import asyncio
import builtins
import dataclasses
import inspect
import threading

import pytest

from flyto_ai.tools import core_tools


try:  # The installed Core, not a stand-in.
    from core.plugin import loader as core_plugin_loader
except ImportError:  # pragma: no cover - exercised on hosts without flyto-core
    core_plugin_loader = None

requires_core = pytest.mark.skipif(
    core_plugin_loader is None,
    reason="flyto-core is not installed in this environment",
)

ENV = core_tools.CORE_EXTENSION_INSTALL_ENV

# A realistic installer failure body. Core returning any of this must not turn
# any of it into envelope content.
PIP_NOISE = {
    "stdout": "Collecting flyto-modules-vision\n  Downloading ...",
    "stderr": "ERROR: Could not find a version that satisfies the requirement",
    "output": "\n".join("pip log line {}".format(i) for i in range(200)),
    "log": "Traceback (most recent call last): ...",
    "command": ["pip", "install", "--index-url", "https://internal/simple"],
}

PIP_NOISE_MARKERS = (
    "pip",
    "Collecting",
    "Traceback",
    "index-url",
    "Could not find",
)


@dataclasses.dataclass
class FakeResult:
    """The fields the host reads out of Core's `ExtensionResult`.

    Read structurally by the adapter, so this stands in for the real class
    without claiming to be it. `TestRealCoreContract` proves the real class
    declares these names.
    """

    ok: bool = True
    code: str = ""
    name: str = ""
    kind: str = ""
    version: str = ""
    previous_version: str = ""
    restart_required: bool = False
    rolled_back: bool = False
    refresh_failed: bool = False


@dataclasses.dataclass
class FakeExtension:
    name: str
    kind: str = ""
    version: str = ""
    installed: bool = True


@dataclasses.dataclass
class FakeKind:
    kind: str
    prefix: str = ""
    entry_point_group: str = ""


class FakeLoader:
    """A plugin loader in the real contract's shape.

    The method names are Core's, not the host's operation names: a loader that
    answered to `install` would have hidden the bug this file now guards.
    `list_extensions` takes no kind argument.
    """

    def __init__(self, calls, extensions=None, install=None, uninstall=None):
        self._calls = calls
        self._extensions = [] if extensions is None else extensions
        self._install = install
        self._uninstall = uninstall

    def list_extensions(self, **kwargs):
        self._calls.append(("list_extensions", kwargs, threading.current_thread()))
        if callable(self._extensions):
            return self._extensions(**kwargs)
        return self._extensions

    def install_extension(self, **kwargs):
        self._calls.append(("install_extension", kwargs, threading.current_thread()))
        if callable(self._install):
            return self._install(**kwargs)
        return FakeResult() if self._install is None else self._install

    def uninstall_extension(self, **kwargs):
        self._calls.append(
            ("uninstall_extension", kwargs, threading.current_thread()),
        )
        if callable(self._uninstall):
            return self._uninstall(**kwargs)
        return FakeResult() if self._uninstall is None else self._uninstall


def _manager(monkeypatch, *, kinds=None, normalize=None, **loader_kwargs):
    """Bind a fake `core.plugin.loader` surface through the adapter's seam."""
    calls = []
    loader = FakeLoader(calls, **loader_kwargs)
    manager = {
        "loader": lambda: loader,
        "kinds": [FakeKind("modules", "flyto-modules-", "flyto.modules")]
        if kinds is None
        else kinds,
        "normalize": normalize or (lambda name: name.strip().lower().replace("_", "-")),
        "result_type": FakeResult,
        "_calls": calls,
    }
    monkeypatch.setattr(core_tools, "_get_core_extension_manager", lambda: manager)
    return manager


def _enable(monkeypatch):
    monkeypatch.setenv(ENV, "true")


def _disable(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def _assert_envelope(result, operation):
    """Every outcome carries exactly the fixed envelope keys."""
    assert isinstance(result, dict)
    assert set(result) == set(core_tools._EXTENSION_ENVELOPE_KEYS)
    assert result["contract"] == core_tools.CORE_EXTENSION_CONTRACT
    assert result["source"] == "flyto-core"
    assert result["operation"] == operation
    for key in (
        "ok", "install_enabled", "restart_required", "rolled_back",
        "refresh_failed",
    ):
        assert isinstance(result[key], bool)
    for key in ("code", "name", "kind", "version", "previous_version"):
        assert isinstance(result[key], str)
    for key in ("extensions", "kinds"):
        assert isinstance(result[key], list)


# ---------------------------------------------------------------------------
# The installed Core, not a fixture
# ---------------------------------------------------------------------------


@requires_core
class TestRealCoreContract:
    def test_core_exports_the_four_names_this_bridge_binds(self):
        assert callable(core_plugin_loader.get_plugin_loader)
        assert callable(core_plugin_loader.normalize_extension_name)
        assert isinstance(core_plugin_loader.EXTENSION_KINDS, (list, tuple))
        assert isinstance(core_plugin_loader.ExtensionResult, type)

    def test_the_adapter_binds_the_installed_core(self, monkeypatch):
        monkeypatch.setattr(core_tools, "_extension_manager_checked", False)
        monkeypatch.setattr(core_tools, "_cached_extension_manager", None)

        manager = core_tools._get_core_extension_manager()

        assert manager is not None
        assert manager["loader"] is core_plugin_loader.get_plugin_loader
        assert manager["normalize"] is core_plugin_loader.normalize_extension_name
        assert manager["result_type"] is core_plugin_loader.ExtensionResult

    def test_extension_result_declares_every_published_field(self):
        """A Core rename must fail here, not silently empty the envelope."""
        result_type = core_plugin_loader.ExtensionResult
        if dataclasses.is_dataclass(result_type):
            declared = {field.name for field in dataclasses.fields(result_type)}
        else:
            declared = set(getattr(result_type, "__annotations__", {}))
        assert declared, "ExtensionResult declares no readable fields"

        missing = [
            field
            for field in core_tools._EXTENSION_RESULT_FIELDS
            if field not in declared
        ]
        assert not missing, (
            "flyto_ai reads ExtensionResult fields Core does not declare: "
            "{}; declared fields are {}".format(sorted(missing), sorted(declared))
        )

    def test_the_plugin_loader_exposes_the_methods_the_adapter_calls(self):
        loader = core_plugin_loader.get_plugin_loader()

        for method in core_tools._EXTENSION_LOADER_METHODS.values():
            assert callable(getattr(loader, method, None)), method

    def test_list_extensions_takes_no_kind_argument(self):
        """The kind filter is host-side because Core's method has no parameter."""
        loader = core_plugin_loader.get_plugin_loader()
        parameters = inspect.signature(loader.list_extensions).parameters

        assert "kind" not in parameters

    def test_extension_kinds_normalize_to_safe_records(self):
        kinds = core_tools._normalize_extension_kinds(
            core_plugin_loader.EXTENSION_KINDS,
        )

        assert kinds, "Core declares no usable extension kinds"
        for record in kinds:
            assert set(record) == set(core_tools._EXTENSION_KIND_FIELDS)
            assert core_tools._SAFE_EXTENSION_KIND.fullmatch(record["kind"])
            assert record["prefix"], "a kind without a prefix selects nothing"
            assert record["entry_point_group"]

    def test_normalize_extension_name_returns_a_safe_identity(self):
        normalized = core_plugin_loader.normalize_extension_name(
            "Flyto_Modules_Vision",
        )

        assert core_tools._safe_extension_token(
            normalized, core_tools._SAFE_EXTENSION_NAME,
        ) is not None

    def test_kinds_against_the_installed_core(self, monkeypatch):
        monkeypatch.setattr(core_tools, "_extension_manager_checked", False)
        monkeypatch.setattr(core_tools, "_cached_extension_manager", None)
        _disable(monkeypatch)

        result = asyncio.run(core_tools.list_core_extension_kinds())

        _assert_envelope(result, "kinds")
        assert result["ok"] is True
        assert result["install_enabled"] is False
        assert result["kinds"]

    def test_list_against_the_installed_core(self, monkeypatch):
        monkeypatch.setattr(core_tools, "_extension_manager_checked", False)
        monkeypatch.setattr(core_tools, "_cached_extension_manager", None)
        _disable(monkeypatch)

        result = asyncio.run(core_tools.list_core_extensions())

        _assert_envelope(result, "list")
        # A readable Core must not validate down to a refusal.
        assert result["code"] != core_tools.EXTENSION_CODE_CORE_UNAVAILABLE
        assert result["code"] != core_tools.EXTENSION_CODE_INVALID_RESULT
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Host-only boundary
# ---------------------------------------------------------------------------


def test_install_tools_never_enter_the_llm_tool_catalog(monkeypatch):
    """A Core that publishes installers must not widen this host's LLM surface."""
    handler = {
        "TOOLS": [
            {"name": "list_modules", "description": "", "inputSchema": {}},
            {"name": "install_extension", "description": "", "inputSchema": {}},
            {"name": "uninstall_extension", "description": "", "inputSchema": {}},
            {"name": "extension_install", "description": "", "inputSchema": {}},
            {"name": "reinstall", "description": "", "inputSchema": {}},
            {"name": "install", "description": "", "inputSchema": {}},
        ],
    }
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)

    names = {tool["name"] for tool in core_tools.get_core_tool_defs()}

    assert "list_modules" in names
    assert names.isdisjoint({
        "install_extension",
        "uninstall_extension",
        "extension_install",
        "reinstall",
        "install",
    })


def test_read_only_installed_reporting_tool_is_not_filtered(monkeypatch):
    """`installed` is a report, not an install verb, so it stays callable."""
    handler = {
        "TOOLS": [
            {"name": "list_installed_modules", "description": "", "inputSchema": {}},
            {"name": "installer_notes", "description": "", "inputSchema": {}},
        ],
    }
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)

    names = {tool["name"] for tool in core_tools.get_core_tool_defs()}

    assert {"list_installed_modules", "installer_notes"} <= names


@pytest.mark.parametrize(
    "name", ["install_extension", "uninstall_extension", "install"],
)
def test_dispatch_refuses_install_tools_even_when_typed_directly(monkeypatch, name):
    """The catalog filter is not the boundary; dispatch refuses too."""
    called = []

    def handler():
        called.append(True)
        return {"TOOLS": []}

    monkeypatch.setattr(core_tools, "_get_mcp_handler", handler)

    result = asyncio.run(core_tools.dispatch_core_tool(name, {"name": "x"}))

    assert result["ok"] is False
    assert "host-only" in result["error"]
    assert called == []


def test_extension_functions_are_not_core_mcp_tools(monkeypatch):
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: {"TOOLS": []})

    names = {tool["name"] for tool in core_tools.get_core_tool_defs()}

    assert names.isdisjoint({
        "install_core_extension",
        "uninstall_core_extension",
        "list_core_extensions",
        "list_core_extension_kinds",
    })


# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------


def test_install_is_disabled_by_default_and_never_reaches_core(monkeypatch):
    _disable(monkeypatch)
    manager = _manager(monkeypatch)

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    _assert_envelope(result, "install")
    assert result["ok"] is False
    assert result["code"] == core_tools.EXTENSION_CODE_INSTALL_DISABLED
    assert result["install_enabled"] is False
    assert manager["_calls"] == []


def test_uninstall_is_disabled_by_default_and_never_reaches_core(monkeypatch):
    _disable(monkeypatch)
    manager = _manager(monkeypatch)

    result = asyncio.run(core_tools.uninstall_core_extension("flyto-modules-vision"))

    _assert_envelope(result, "uninstall")
    assert result["code"] == core_tools.EXTENSION_CODE_INSTALL_DISABLED
    assert manager["_calls"] == []


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", " "])
def test_non_truthy_opt_in_values_keep_mutation_disabled(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    manager = _manager(monkeypatch)

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["code"] == core_tools.EXTENSION_CODE_INSTALL_DISABLED
    assert manager["_calls"] == []


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " true "])
def test_truthy_opt_in_values_enable_mutation(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    _manager(monkeypatch, install=FakeResult(ok=True, code="installed"))

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["ok"] is True
    assert result["code"] == "installed"
    assert result["install_enabled"] is True


def test_disabled_gate_answers_before_validating_the_request(monkeypatch):
    """A disabled host gives one answer to every input, valid or not."""
    _disable(monkeypatch)
    manager = _manager(monkeypatch)

    result = asyncio.run(core_tools.install_core_extension("--index-url=https://x"))

    assert result["code"] == core_tools.EXTENSION_CODE_INSTALL_DISABLED
    assert manager["_calls"] == []


def test_read_operations_do_not_need_the_opt_in(monkeypatch):
    """Listing is not mutation; it stays available on an un-opted-in host."""
    _disable(monkeypatch)
    _manager(monkeypatch, extensions=[FakeExtension("flyto-modules-vision")])

    listed = asyncio.run(core_tools.list_core_extensions())
    kinds = asyncio.run(core_tools.list_core_extension_kinds())

    assert listed["ok"] is True
    assert kinds["ok"] is True
    assert listed["install_enabled"] is False


# ---------------------------------------------------------------------------
# Absent Core
# ---------------------------------------------------------------------------


def test_absent_core_is_reported_not_answered_as_empty(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(core_tools, "_get_core_extension_manager", lambda: None)

    listed = asyncio.run(core_tools.list_core_extensions())
    kinds = asyncio.run(core_tools.list_core_extension_kinds())
    installed = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))
    removed = asyncio.run(core_tools.uninstall_core_extension("flyto-modules-vision"))

    for result, operation in (
        (listed, "list"),
        (kinds, "kinds"),
        (installed, "install"),
        (removed, "uninstall"),
    ):
        _assert_envelope(result, operation)
        assert result["ok"] is False
        assert result["code"] == core_tools.EXTENSION_CODE_CORE_UNAVAILABLE

    # "Cannot answer" is not "nothing installed".
    assert listed["extensions"] == []
    assert kinds["kinds"] == []


def test_a_core_without_the_plugin_loader_binds_to_nothing(monkeypatch):
    monkeypatch.setattr(core_tools, "_extension_manager_checked", False)
    monkeypatch.setattr(core_tools, "_cached_extension_manager", None)

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "core.plugin.loader":
            raise ImportError("no such module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    assert core_tools._get_core_extension_manager() is None


# ---------------------------------------------------------------------------
# list / kinds
# ---------------------------------------------------------------------------


def test_list_reads_cores_plain_list_and_preserves_names(monkeypatch):
    _manager(monkeypatch, extensions=[
        FakeExtension("flyto-modules-vision", "Modules", "2.3.0", True),
        FakeExtension("flyto_modules_ops"),
        FakeExtension("flyto-modules-legacy", installed=False),
    ])

    result = asyncio.run(core_tools.list_core_extensions())

    _assert_envelope(result, "list")
    assert result["ok"] is True
    assert result["extensions"] == [
        {
            "name": "flyto-modules-vision",
            "kind": "modules",
            "version": "2.3.0",
            "installed": True,
        },
        {"name": "flyto_modules_ops", "kind": "", "version": "", "installed": True},
        {
            "name": "flyto-modules-legacy",
            "kind": "",
            "version": "",
            "installed": False,
        },
    ]


def test_the_kind_filter_is_applied_host_side(monkeypatch):
    """Core's `list_extensions` takes no kind argument, so the host narrows."""
    manager = _manager(monkeypatch, extensions=[
        FakeExtension("flyto-modules-vision", "modules"),
        FakeExtension("flyto-recipes-ops", "recipes"),
    ])

    everything = asyncio.run(core_tools.list_core_extensions())
    narrowed = asyncio.run(core_tools.list_core_extensions("modules"))

    # Core is called identically both times, with no filter argument.
    assert [call[1] for call in manager["_calls"]] == [{}, {}]
    assert [record["name"] for record in everything["extensions"]] == [
        "flyto-modules-vision", "flyto-recipes-ops",
    ]
    assert [record["name"] for record in narrowed["extensions"]] == [
        "flyto-modules-vision",
    ]
    assert narrowed["kind"] == "modules"


def test_a_kind_filter_matching_nothing_is_an_empty_list_not_a_failure(
    monkeypatch,
):
    _manager(monkeypatch, extensions=[FakeExtension("flyto-modules-vision")])

    result = asyncio.run(core_tools.list_core_extensions("recipes"))

    assert result["ok"] is True
    assert result["extensions"] == []


def test_list_rejects_an_unsafe_kind_before_calling_core(monkeypatch):
    manager = _manager(monkeypatch)

    result = asyncio.run(core_tools.list_core_extensions("../../etc"))

    _assert_envelope(result, "list")
    assert result["code"] == core_tools.EXTENSION_CODE_INVALID_REQUEST
    assert manager["_calls"] == []


@pytest.mark.parametrize(
    "extensions",
    [
        "not-a-list",
        [FakeExtension("-rf /")],
        [FakeExtension("ok", kind="not a kind")],
        [FakeExtension("ok", version="1.0 ; rm -rf /")],
        [FakeExtension("dupe"), FakeExtension("dupe")],
        ["flyto-modules-vision"],
    ],
)
def test_list_fails_closed_on_a_malformed_core_answer(monkeypatch, extensions):
    _manager(monkeypatch, extensions=extensions)

    result = asyncio.run(core_tools.list_core_extensions())

    _assert_envelope(result, "list")
    assert result["ok"] is False
    assert result["code"] == core_tools.EXTENSION_CODE_INVALID_RESULT
    assert result["extensions"] == []


def test_a_raising_list_is_a_core_error(monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("pip install failed\n" + PIP_NOISE["output"])

    _manager(monkeypatch, extensions=explode)

    result = asyncio.run(core_tools.list_core_extensions())

    assert result["code"] == core_tools.EXTENSION_CODE_CORE_ERROR


def test_kinds_are_records_and_are_whatever_core_declares(monkeypatch):
    """The adapter is generic: it holds no kind taxonomy of its own."""
    _manager(monkeypatch, kinds=[
        FakeKind("modules", "flyto-modules-", "flyto.modules"),
        FakeKind("recipes", "flyto-recipes-"),
        FakeKind("some-future-kind"),
    ])

    result = asyncio.run(core_tools.list_core_extension_kinds())

    _assert_envelope(result, "kinds")
    assert result["ok"] is True
    assert [record["kind"] for record in result["kinds"]] == [
        "modules", "recipes", "some-future-kind",
    ]
    assert result["kinds"][0]["prefix"] == "flyto-modules-"
    assert result["kinds"][0]["entry_point_group"] == "flyto.modules"
    assert result["kinds"][2]["prefix"] == ""


def test_kinds_reads_the_constant_without_touching_the_loader(monkeypatch):
    manager = _manager(monkeypatch)

    asyncio.run(core_tools.list_core_extension_kinds())

    assert manager["_calls"] == []


@pytest.mark.parametrize(
    "kinds",
    [
        "modules",
        ["modules"],
        [FakeKind("modules"), FakeKind("modules")],
        [FakeKind("BAD KIND")],
    ],
)
def test_kinds_fail_closed_on_a_malformed_core_answer(monkeypatch, kinds):
    _manager(monkeypatch, kinds=kinds)

    result = asyncio.run(core_tools.list_core_extension_kinds())

    assert result["ok"] is False
    assert result["code"] == core_tools.EXTENSION_CODE_INVALID_RESULT
    assert result["kinds"] == []


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


def test_install_normalizes_through_core_before_calling_it(monkeypatch):
    _enable(monkeypatch)
    manager = _manager(monkeypatch)

    result = asyncio.run(core_tools.install_core_extension("Flyto_Modules_Vision"))

    assert manager["_calls"][0][1] == {
        "name": "flyto-modules-vision",
        "version": None,
        "upgrade": False,
    }
    assert result["name"] == "flyto-modules-vision"


def test_install_forwards_version_and_upgrade(monkeypatch):
    _enable(monkeypatch)
    manager = _manager(monkeypatch)

    asyncio.run(
        core_tools.install_core_extension(
            "flyto-modules-vision", version="2.3.0", upgrade=True,
        ),
    )

    assert manager["_calls"][0][1] == {
        "name": "flyto-modules-vision",
        "version": "2.3.0",
        "upgrade": True,
    }


def test_uninstall_passes_only_the_name(monkeypatch):
    _enable(monkeypatch)
    manager = _manager(monkeypatch)

    asyncio.run(core_tools.uninstall_core_extension("flyto-modules-vision"))

    assert manager["_calls"] == [
        (
            "uninstall_extension",
            {"name": "flyto-modules-vision"},
            manager["_calls"][0][2],
        ),
    ]


def test_install_publishes_cores_code_name_versions_and_flags(monkeypatch):
    _enable(monkeypatch)
    _manager(monkeypatch, install=FakeResult(
        ok=True,
        code="installed",
        name="flyto-modules-vision",
        kind="modules",
        version="2.3.0",
        previous_version="2.2.9",
        restart_required=True,
        refresh_failed=True,
        rolled_back=False,
    ))

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    _assert_envelope(result, "install")
    assert result["ok"] is True
    assert result["code"] == "installed"
    assert result["name"] == "flyto-modules-vision"
    assert result["kind"] == "modules"
    assert result["version"] == "2.3.0"
    assert result["previous_version"] == "2.2.9"
    assert result["restart_required"] is True
    assert result["refresh_failed"] is True
    assert result["rolled_back"] is False


def test_an_extra_status_alias_cannot_override_cores_explicit_failure(monkeypatch):
    """ExtensionResult success is its exact ``ok`` field, not a module alias."""
    _enable(monkeypatch)

    @dataclasses.dataclass
    class ResultWithAlias(FakeResult):
        status: str = "success"

    _manager(monkeypatch, install=ResultWithAlias(
        ok=False,
        code="install_failed",
        name="flyto-modules-vision",
        kind="modules",
    ))

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["ok"] is False
    assert result["code"] == "install_failed"


def test_a_failed_install_reports_cores_code_and_rollback(monkeypatch):
    _enable(monkeypatch)
    _manager(monkeypatch, install=FakeResult(
        ok=False,
        code="build_failed",
        name="flyto-modules-vision",
        rolled_back=True,
    ))

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["ok"] is False
    assert result["code"] == "build_failed"
    assert result["rolled_back"] is True


def test_install_enabled_reports_the_host_opt_in_not_a_core_field(monkeypatch):
    """Core does not report installability; this field is the host's own state."""
    _enable(monkeypatch)
    _manager(monkeypatch, install=FakeResult(ok=False, code="build_failed"))

    enabled = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    _disable(monkeypatch)
    disabled = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert enabled["install_enabled"] is True
    assert disabled["install_enabled"] is False


@pytest.mark.parametrize("flag", core_tools._EXTENSION_RESULT_FLAG_FIELDS)
def test_an_unreported_flag_is_never_read_as_yes(monkeypatch, flag):
    _enable(monkeypatch)

    @dataclasses.dataclass
    class SparseResult:
        ok: bool = True
        code: str = "installed"

    _manager(monkeypatch, install=SparseResult())

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result[flag] is False


@pytest.mark.parametrize(
    "code",
    [
        "",
        "   ",
        "Collecting flyto-modules-vision from https://internal/simple",
        "x" * 200,
    ],
)
def test_an_unsafe_core_code_degrades_to_a_host_code(monkeypatch, code):
    _enable(monkeypatch)
    _manager(monkeypatch, install=FakeResult(ok=False, code=code))

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["code"] == core_tools.EXTENSION_CODE_CORE_ERROR


@pytest.mark.parametrize(
    "name",
    [
        "",
        "-e .",
        "--index-url=https://internal/simple",
        "../../etc/passwd",
        "flyto-modules-vision; rm -rf /",
        "https://example.invalid/pkg.tar.gz",
        "flyto modules vision",
        "x" * 200,
        None,
        123,
    ],
)
def test_an_unsafe_requested_name_never_reaches_core(monkeypatch, name):
    _enable(monkeypatch)
    manager = _manager(monkeypatch)

    installed = asyncio.run(core_tools.install_core_extension(name))
    removed = asyncio.run(core_tools.uninstall_core_extension(name))

    for result, operation in ((installed, "install"), (removed, "uninstall")):
        _assert_envelope(result, operation)
        assert result["ok"] is False
        assert result["code"] == core_tools.EXTENSION_CODE_INVALID_REQUEST
    assert manager["_calls"] == []


@pytest.mark.parametrize("version", ["-rf", "2.3.0 ; rm -rf /", "x" * 200, 2.3])
def test_an_unsafe_version_never_reaches_core(monkeypatch, version):
    _enable(monkeypatch)
    manager = _manager(monkeypatch)

    result = asyncio.run(
        core_tools.install_core_extension("flyto-modules-vision", version=version),
    )

    assert result["code"] == core_tools.EXTENSION_CODE_INVALID_REQUEST
    assert manager["_calls"] == []


def test_a_non_boolean_upgrade_never_reaches_core(monkeypatch):
    _enable(monkeypatch)
    manager = _manager(monkeypatch)

    result = asyncio.run(
        core_tools.install_core_extension("flyto-modules-vision", upgrade="yes"),
    )

    assert result["code"] == core_tools.EXTENSION_CODE_INVALID_REQUEST
    assert manager["_calls"] == []


def test_an_unsafe_normalized_name_is_a_core_result_failure(monkeypatch):
    _enable(monkeypatch)
    manager = _manager(monkeypatch, normalize=lambda name: "--index-url=x")

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["code"] == core_tools.EXTENSION_CODE_INVALID_RESULT
    assert manager["_calls"] == []


def test_a_raising_normalize_is_a_core_error(monkeypatch):
    _enable(monkeypatch)

    def explode(name):
        raise RuntimeError("boom")

    manager = _manager(monkeypatch, normalize=explode)

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["code"] == core_tools.EXTENSION_CODE_CORE_ERROR
    assert manager["_calls"] == []


def test_an_unreadable_result_object_is_invalid(monkeypatch):
    _enable(monkeypatch)
    _manager(monkeypatch, install=lambda **kwargs: "installed fine")

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    _assert_envelope(result, "install")
    assert result["ok"] is False
    assert result["code"] == core_tools.EXTENSION_CODE_INVALID_RESULT


def test_a_missing_loader_method_is_a_core_error(monkeypatch):
    _enable(monkeypatch)

    class Empty:
        pass

    manager = {
        "loader": Empty,
        "kinds": [],
        "normalize": lambda name: name,
        "result_type": FakeResult,
    }
    monkeypatch.setattr(core_tools, "_get_core_extension_manager", lambda: manager)

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert result["code"] == core_tools.EXTENSION_CODE_CORE_ERROR


# ---------------------------------------------------------------------------
# No installer output, ever
# ---------------------------------------------------------------------------


def _assert_no_installer_output(result):
    rendered = core_tools.json_dumps(result)
    for marker in PIP_NOISE_MARKERS:
        assert marker not in rendered, rendered


def test_a_result_carrying_installer_output_never_leaks_it(monkeypatch):
    """Extra fields on Core's result are dropped by the fixed envelope."""
    _enable(monkeypatch)

    @dataclasses.dataclass
    class NoisyResult:
        ok: bool = False
        code: str = "build_failed"
        name: str = "flyto-modules-vision"
        version: str = ""
        previous_version: str = ""
        restart_required: bool = False
        rolled_back: bool = True
        refresh_failed: bool = False
        stdout: str = PIP_NOISE["stdout"]
        stderr: str = PIP_NOISE["stderr"]
        output: str = PIP_NOISE["output"]
        log: str = PIP_NOISE["log"]

    _manager(monkeypatch, install=NoisyResult())

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    _assert_envelope(result, "install")
    assert result["code"] == "build_failed"
    assert result["rolled_back"] is True
    _assert_no_installer_output(result)


def test_a_raising_install_never_leaks_the_installer_traceback(monkeypatch):
    _enable(monkeypatch)

    def explode(**kwargs):
        raise RuntimeError(
            "pip install failed\n" + PIP_NOISE["output"] + "\n" + PIP_NOISE["log"],
        )

    _manager(monkeypatch, install=explode)

    result = asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    _assert_envelope(result, "install")
    assert result["ok"] is False
    assert result["code"] == core_tools.EXTENSION_CODE_CORE_ERROR
    _assert_no_installer_output(result)


def test_a_raising_install_is_logged_without_installer_output(monkeypatch, caplog):
    _enable(monkeypatch)

    def explode(**kwargs):
        raise RuntimeError("pip install failed\n" + PIP_NOISE["output"])

    _manager(monkeypatch, install=explode)

    with caplog.at_level("WARNING", logger=core_tools.logger.name):
        asyncio.run(core_tools.install_core_extension("flyto-modules-vision"))

    assert "RuntimeError" in caplog.text
    assert "pip log line" not in caplog.text


def test_core_kind_provenance_is_bounded(monkeypatch):
    _manager(monkeypatch, kinds=[FakeKind("modules", "x" * 500, "y" * 500)])

    result = asyncio.run(core_tools.list_core_extension_kinds())

    assert len(result["kinds"][0]["prefix"]) == core_tools._MAX_EXTENSION_LABEL
    assert (
        len(result["kinds"][0]["entry_point_group"])
        == core_tools._MAX_EXTENSION_LABEL
    )


# ---------------------------------------------------------------------------
# Off-loop execution
# ---------------------------------------------------------------------------


def test_loader_calls_run_off_the_event_loop(monkeypatch):
    """A blocking install must not stall every other task on the loop."""
    _enable(monkeypatch)
    manager = _manager(monkeypatch)

    async def scenario():
        loop_thread = threading.current_thread()
        await core_tools.list_core_extensions()
        await core_tools.install_core_extension("flyto-modules-vision")
        await core_tools.uninstall_core_extension("flyto-modules-vision")
        return loop_thread

    loop_thread = asyncio.run(scenario())

    assert [call[0] for call in manager["_calls"]] == [
        "list_extensions", "install_extension", "uninstall_extension",
    ]
    for _method, _kwargs, thread in manager["_calls"]:
        assert thread is not loop_thread


def test_a_slow_install_does_not_block_other_tasks(monkeypatch):
    _enable(monkeypatch)
    started = threading.Event()
    release = threading.Event()

    def slow(**kwargs):
        started.set()
        # Bounded, and released by the concurrent task below.
        release.wait(5)
        return FakeResult(ok=True, name="flyto-modules-vision")

    _manager(monkeypatch, install=slow)

    async def scenario():
        task = asyncio.ensure_future(
            core_tools.install_core_extension("flyto-modules-vision"),
        )
        while not started.is_set():
            await asyncio.sleep(0.01)
        # The loop is still live while Core blocks in its worker thread.
        release.set()
        return await task

    result = asyncio.run(scenario())

    assert result["ok"] is True
