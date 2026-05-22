"""Dispatcher — runs inspectors, selects components, executes them.

The dispatcher is the *only* code in the extraction pipeline that
knows about composition.  Per docs/extraction_pipeline.md:

  1. Run inspectors in dependency order; accumulate properties.
  2. Filter components by ``preconditions(props)``.
  3. Topologically sort selected components by their
     ``requires_outputs`` / ``produces_outputs`` declarations so
     components that consume another's output run after it.
  4. Construct each component's :class:`ComponentInstance` lazily
     and call ``run(state)``.
  5. The treatment_assembler is the terminal stratum and writes
     ``state.treatments``.

The ``Dispatcher`` is itself stateless — one instance can extract
many docs sequentially, reusing the same inspector + component
catalogs.  ``extract(doc)`` returns the per-doc Treatment list.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .catalog import MemoryCatalog
from .interfaces import Component, Inspector
from .state import PipelineState

log = logging.getLogger(__name__)


class DispatcherError(Exception):
    """Raised when the dispatcher can't construct a valid plan."""


class Dispatcher:
    """Composes inspectors + components into a per-doc extraction.

    Args:
        inspectors: A populated :class:`MemoryCatalog[Inspector]`.
        components: A populated :class:`MemoryCatalog[Component]`.
        config: Pipeline config snapshot (env_config dict).  Forwarded
            to :class:`PipelineState`; components read shared service
            handles + per-pipeline settings from it.
        couchdb_db: Optional live CouchDB DB handle for attachment
            fetches.  When unset, components only see attachments
            pre-seeded in ``doc['_attachments']`` (test mode).
        redis_client: Optional Redis handle for components that need
            it.
    """

    def __init__(
        self,
        inspectors: MemoryCatalog[Inspector],
        components: MemoryCatalog[Component],
        config: Optional[Dict[str, Any]] = None,
        couchdb_db: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self._inspectors = inspectors
        self._components = components
        self._config = config or {}
        self._couchdb_db = couchdb_db
        self._redis_client = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, doc: Dict[str, Any]) -> List[Any]:
        """Run the pipeline on a single doc; return its Treatments.

        The returned list is also available as ``state.treatments``
        on the state object the dispatcher built (useful in tests
        that want to inspect ``state.props`` etc.).  ``extract_state``
        is the lower-level entry that returns the state directly.
        """
        return self.extract_state(doc).treatments

    def extract_state(self, doc: Dict[str, Any]) -> PipelineState:
        """Run the pipeline on a single doc; return the final state."""
        state = PipelineState(
            doc=doc,
            config=self._config,
            couchdb_db=self._couchdb_db,
            redis_client=self._redis_client,
        )

        # Phase 1 — inspectors.
        self._run_inspectors(state)

        # Phase 2 — component selection.
        selected = self._select_components(state.props)

        # Phase 3 — topological sort by output dependencies.
        plan = self._topological_strata(selected)

        # Phase 4 — execute in stratum order.
        for stratum in plan:
            for descriptor in stratum:
                try:
                    instance = descriptor.create_instance()
                    instance.run(state)
                except Exception:  # noqa: BLE001
                    log.exception(
                        "component %s raised on doc %s",
                        descriptor.name, state.doc.get("_id"),
                    )

        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_inspectors(self, state: PipelineState) -> None:
        """Run every inspector whose ``requires`` set is satisfied,
        repeating until no more can run.

        We don't pre-sort inspectors topologically because the
        ``requires`` declaration is over *property names* (which any
        inspector may add) rather than over other inspectors by
        name.  A fixed-point loop is the simplest correct approach
        and the inspector count is small.
        """
        remaining = list(self._inspectors.all_objects())
        while True:
            progress = False
            still_pending: List[Inspector] = []
            for inspector in remaining:
                if inspector.requires.issubset(state.props.keys()):
                    new_props = inspector.inspect(state.doc, state.props)
                    state.props.update(new_props)
                    progress = True
                else:
                    still_pending.append(inspector)
            remaining = still_pending
            if not remaining or not progress:
                break

    def _select_components(
        self, props: Dict[str, Any],
    ) -> List[Component]:
        return [
            c for c in self._components.all_objects()
            if c.preconditions(props)
        ]

    def _topological_strata(
        self, components: List[Component],
    ) -> List[List[Component]]:
        """Return a list of strata; each stratum is a set of components
        with no dependency on each other.

        Dependencies are inferred from ``requires_outputs`` /
        ``produces_outputs``: a component that requires output X
        runs after any component that produces X.

        Raises :class:`DispatcherError` if no valid order exists
        (cycle, or a requires_outputs that no selected component
        produces).
        """
        producer_of: Dict[str, List[Component]] = defaultdict(list)
        for c in components:
            for out in c.produces_outputs:
                producer_of[out].append(c)

        remaining: Set[Component] = set(components)
        plan: List[List[Component]] = []

        while remaining:
            stratum: List[Component] = []
            for c in list(remaining):
                # A component is ready when every output it requires
                # is produced by a component already in an earlier
                # stratum (or no component in the selection produces
                # it, in which case the dispatcher silently treats
                # that as "no producer needed" — the assembler's
                # ``requires_outputs={treatment_labels}`` is still
                # honoured naturally because the labeler that
                # produces ``treatment_labels`` is in an earlier
                # stratum).
                pending_deps = [
                    out for out in c.requires_outputs
                    if any(
                        prod in remaining
                        for prod in producer_of.get(out, [])
                    )
                ]
                if not pending_deps:
                    stratum.append(c)

            if not stratum:
                raise DispatcherError(
                    "no valid topological order — likely a circular "
                    "requires_outputs dependency among components: "
                    + ", ".join(c.name for c in remaining)
                )

            plan.append(stratum)
            remaining.difference_update(stratum)

        return plan

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_default_catalogs(
        cls,
        config: Optional[Dict[str, Any]] = None,
        couchdb_db: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> "Dispatcher":
        """Construct a Dispatcher with the standard skol catalogs.

        Loads ``skol_classifier/extraction/inspectors/`` and
        ``skol_classifier/extraction/components/`` via
        :meth:`MemoryCatalog.load`.
        """
        from pathlib import Path

        inspector_catalog: MemoryCatalog[Inspector] = MemoryCatalog()
        component_catalog: MemoryCatalog[Component] = MemoryCatalog()

        here = Path(__file__).parent
        inspector_catalog.load(here / "inspectors")
        component_catalog.load(here / "components")

        return cls(
            inspectors=inspector_catalog,
            components=component_catalog,
            config=config,
            couchdb_db=couchdb_db,
            redis_client=redis_client,
        )
