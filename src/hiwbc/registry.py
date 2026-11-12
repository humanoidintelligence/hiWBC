"""Controller and profile registries.

Two levels, both by name:

* **controllers** (``register_controller`` / ``make_controller``): a controller class
  with every parameter passed in. Tests and the guides use this.
* **profiles** (``register_profile`` / ``load``): ``<controller>/<robot>``, the trained
  parameters for one robot plus how to load the checkpoint. ``WbcRobot`` uses this.

Profile modules live in ``hiwbc/configs/<controller>/<robot>.py`` and are imported
only when ``load()`` asks for them, so an unused controller never pulls in its model
runtime (torch, onnxruntime).
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .controller import Controller

_REGISTRY: dict[str, Callable[..., Controller]] = {}


def register_controller(name: str) -> Callable[[Callable[..., Controller]], Callable[..., Controller]]:
    """Decorator: register a controller factory under ``name``."""

    def deco(factory: Callable[..., Controller]) -> Callable[..., Controller]:
        if name in _REGISTRY:
            raise ValueError(f"controller {name!r} already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def make_controller(name: str, **kwargs) -> Controller:
    """Construct a registered controller by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown controller {name!r}; available: {available_controllers()}")
    return _REGISTRY[name](**kwargs)


def available_controllers() -> list[str]:
    """Names of all registered controllers."""
    return sorted(_REGISTRY)


# ----- profiles ----------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    """One ``<controller>/<robot>`` profile.

    ``build(variant, options)`` returns the controller. ``variants`` maps a variant
    name to the profile's per-variant data (handed to ``build``); a profile with one
    configuration has a single ``"default"`` variant. ``default_options`` lists every
    option ``load()`` accepts besides ``variant``, with its default; any other key is
    an error.
    """

    name: str
    build: Callable[[Any, dict[str, Any]], Controller]
    variants: Mapping[str, Any]
    default_options: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def default_variant(self) -> str:
        return "default" if "default" in self.variants else next(iter(self.variants))


_PROFILES: dict[str, Profile] = {}
_CONFIGS_PACKAGE = __name__.rsplit(".", 1)[0] + ".configs"


def register_profile(
    name: str,
    *,
    build: Callable[[Any, dict[str, Any]], Controller],
    variants: Mapping[str, Any],
    default_options: Mapping[str, Any] | None = None,
    description: str = "",
) -> Profile:
    """Register a profile ``<controller>/<robot>``. Call it from a ``configs/`` module."""
    if name.count("/") != 1 or not all(name.split("/")):
        raise ValueError(f"profile name {name!r} must be '<controller>/<robot>'")
    if name in _PROFILES:
        raise ValueError(f"profile {name!r} already registered")
    if not variants:
        raise ValueError(f"profile {name!r} needs at least one variant")
    options = dict(default_options or {})
    if "variant" in options:
        raise ValueError("'variant' is implied; do not list it in default_options")
    profile = Profile(name=name, build=build, variants=dict(variants),
                      default_options=options, description=description)
    _PROFILES[name] = profile
    return profile


def _import_profile(name: str) -> None:
    controller, robot = name.split("/")
    module = f"{_CONFIGS_PACKAGE}.{controller}.{robot}"
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:
        # Only "no such profile module" maps to KeyError; a profile whose own
        # import fails (missing torch, a typo) surfaces as is.
        if e.name not in (module, f"{_CONFIGS_PACKAGE}.{controller}"):
            raise


def get_profile(name: str) -> Profile:
    """Look up a profile, importing its ``configs/`` module on first use."""
    if name not in _PROFILES and name.count("/") == 1:
        _import_profile(name)
    if name not in _PROFILES:
        raise KeyError(f"unknown profile {name!r}; available: {available_profiles()}")
    return _PROFILES[name]


def load(profile: str, **options) -> Controller:
    """Build the controller of ``profile`` (``<controller>/<robot>``).

    ``options`` are the profile's ``default_options`` keys plus ``variant``; unknown
    keys raise ``TypeError``. Loading the checkpoint happens here, so a missing file
    or a wrong joint count fails now, not on the first ``step()``.

    ``runtime="remote"`` with ``endpoint="tcp://host:port"`` runs the model forward
    on ``python -m hiwbc.serve`` instead (:func:`hiwbc.remote.make_remote`).
    The checkpoints must still be on this machine: the controller is built from
    them, then its model callables are swapped for remote ones, and the server
    checks that it runs the same profile and variant. ``remote_timeout_s``
    (default 15 ms) bounds each per-tick call.
    """
    p = get_profile(profile)
    runtime = options.pop("runtime", "local")
    endpoint = options.pop("endpoint", None)
    remote_timeout_s = options.pop("remote_timeout_s", 0.015)
    if runtime not in ("local", "remote"):
        raise ValueError(f"runtime must be 'local' or 'remote', got {runtime!r}")
    if (runtime == "remote") != (endpoint is not None):
        raise ValueError("runtime='remote' needs endpoint='tcp://host:port' (and only then)")
    unknown = sorted(set(options) - set(p.default_options) - {"variant"})
    if unknown:
        raise TypeError(
            f"profile {profile!r} got unknown option(s) {unknown}; "
            f"accepted: {sorted(['variant', 'runtime', 'endpoint', 'remote_timeout_s', *p.default_options])}"
        )
    variant = options.pop("variant", None) or p.default_variant
    if variant not in p.variants:
        raise ValueError(f"profile {profile!r} has no variant {variant!r}; available: {sorted(p.variants)}")
    resolved = {**p.default_options, **options}
    ctrl = p.build(p.variants[variant], resolved)
    ctrl.profile = profile
    ctrl.variant = variant
    if runtime == "remote":
        from .remote import make_remote

        make_remote(ctrl, endpoint, timeout_s=remote_timeout_s)
    return ctrl


def available_profiles() -> list[str]:
    """All profile names shipped in ``hiwbc/configs`` plus any registered at runtime.

    Lists module names without importing them, so it pulls in no model runtime.
    """
    pkg = importlib.import_module(_CONFIGS_PACKAGE)
    names = set(_PROFILES)
    for ctrl in pkgutil.iter_modules(pkg.__path__):
        if not ctrl.ispkg:
            continue
        sub = importlib.import_module(f"{_CONFIGS_PACKAGE}.{ctrl.name}")
        for robot in pkgutil.iter_modules(sub.__path__):
            if not robot.name.startswith("_"):
                names.add(f"{ctrl.name}/{robot.name}")
    return sorted(names)
