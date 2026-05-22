"""Abstract interfaces for the skol extraction pipeline.

Three concrete bases:

* :class:`Inspector` — a cheap pure function that reads the doc and
  returns a small dict of properties.  Inspectors don't mutate the
  pipeline state; they only populate the property dict.

* :class:`Component` — a *descriptor* that lives in the
  :class:`ComponentCatalog` and knows which conditions it requires
  (over inspector-produced properties) and what it produces (text,
  spans, section labels, treatments).  Components are cheap to
  register; they don't load heavy state.

* :class:`ComponentInstance` — the *runtime executor* paired with a
  Component.  Constructed lazily by the dispatcher just before
  ``run()``; may hold loaded models, REST clients, or Spark sessions.

Four ``Component`` subtypes match the four categories of output
discussed in docs/extraction_pipeline.md: ``TextProducer``,
``EntityDetector``, ``SectionLabeler``, ``Assembler``.

All concrete components mix in :class:`CatalogElementMixin` to supply
``name`` and ``tags`` for the catalog.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, Type


class Inspector(ABC):
    """Computes document properties; produces no side-effects."""

    #: Property names this inspector needs available before running.
    #: Defaulting to an empty set means it can run first.
    requires: FrozenSet[str] = frozenset()

    @abstractmethod
    def inspect(
        self,
        doc: Dict[str, Any],
        props: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a dict of new properties to merge into ``props``.

        Args:
            doc: The CouchDB document (dict-like).
            props: The property dict accumulated so far (read-only).

        Returns:
            A dict of new properties to add.  Existing keys are not
            overwritten by the dispatcher; an Inspector that needs to
            override a previously-set property must use a different
            key.
        """


class ComponentInstance(ABC):
    """Runtime executor for a Component.

    Constructed lazily by the dispatcher.  May hold heavy state (loaded
    models, REST sessions, Spark contexts) — the descriptor pattern
    means the catalog can list components without loading them.
    """

    @abstractmethod
    def run(self, state: "PipelineState") -> None:  # noqa: F821
        """Mutate ``state`` in place; produce this component's outputs."""


class Component(ABC):
    """Descriptor — cheap, registered in :class:`ComponentCatalog`.

    Subclasses declare:
      * ``requires_props``: inspector-produced property names needed
        for ``preconditions()`` to potentially return True.
      * ``requires_outputs``: other components' output kinds this
        component reads from ``PipelineState`` (used by the dispatcher
        for topological ordering).
      * ``produces_outputs``: what this component contributes to
        ``PipelineState`` (also used for topological ordering).
      * ``instance_constructor``: the :class:`ComponentInstance`
        subclass that does the work.

    ``preconditions(props)`` is called per-document by the dispatcher
    to decide whether to instantiate + run this component.
    """

    #: Inspector-produced property names this component reads.
    requires_props: FrozenSet[str] = frozenset()
    #: Other components' output kinds this component depends on.
    requires_outputs: FrozenSet[str] = frozenset()
    #: Output kinds this component contributes to PipelineState.
    produces_outputs: FrozenSet[str] = frozenset()
    #: ComponentInstance subclass; concrete components override this.
    instance_constructor: Type[ComponentInstance]

    @abstractmethod
    def preconditions(self, props: Dict[str, Any]) -> bool:
        """Return ``True`` iff this component should run for this doc."""

    def create_instance(self, **kwargs: Any) -> ComponentInstance:
        """Construct the paired :class:`ComponentInstance`.

        Default factory passes ``**kwargs`` through.  Subclasses with
        custom instance-init signatures should override.
        """
        return self.instance_constructor(**kwargs)


# Four output-category subtypes.  Functional alias — no extra methods
# yet; presence on the MRO is what the dispatcher (and tag-based
# catalog queries) look for.

class TextProducer(Component):
    """Components that produce text (OCR, plaintext extraction)."""


class EntityDetector(Component):
    """Components that produce ``List[Span]`` contributions."""


class SectionLabeler(Component):
    """Components that produce per-passage ``TaggedBlock`` contributions."""


class Assembler(Component):
    """Component that consumes labels + spans and produces Treatments."""
