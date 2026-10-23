"""Modal definitions for the treatments_to_json Modal backend.

Thin wrappers: each Modal function calls the tested logic in
``treatments_to_structured.modal_models``.  Run ephemerally with
``app.run()`` from local commands (bin/prepare_modal_models); nothing
is deployed.  See docs/modal_backend_design.md.

Tranche 1: CPU-only model preparation.  Tranche 2: the vLLM GPU
class ``JsonTranslator`` and ``run_translator``.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import modal

from treatments_to_structured.modal_inference import (
    CACHE_MOUNT,
    CACHE_VOLUME_NAME,
    CONTAINER_CORES,
    CONTAINER_MEMORY_GIB,
    DTYPE,
    MAX_MODEL_LEN,
    SCALEDOWN_WINDOW_S,
    VLLM_BASE_IMAGE,
    VLLM_VERSION,
    cache_env,
    json_grammar,
    raw_from_vllm,
    sampling_options,
)
from treatments_to_structured.modal_models import (
    MODEL_ID,
    MODEL_REVISION,
    PREPARE_RESOURCES,
    VOLUME_MOUNT,
    VOLUME_NAME,
    check,
    is_prepared,
    model_dir,
    prepare,
)

APP_NAME = 'skol-treatments-json'
HUGGINGFACE_HUB_VERSION = '0.36.2'

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

prepare_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(f'huggingface_hub=={HUGGINGFACE_HUB_VERSION}')
    .add_local_python_source('treatments_to_structured')
)

_cpu_function = app.function(
    image=prepare_image,
    volumes={VOLUME_MOUNT: volume},
    cpu=PREPARE_RESOURCES['cpu'],
    memory=PREPARE_RESOURCES['memory_mib'],
    timeout=PREPARE_RESOURCES['timeout_s'],
)


@_cpu_function
def check_models(model_id: str, revision: str) -> Dict[str, Any]:
    """Report whether the weights are on the Volume; download nothing."""
    return check(Path(VOLUME_MOUNT), model_id, revision)


@_cpu_function
def prepare_models(model_id: str, revision: str) -> str:
    """Download the weights into the Volume unless already prepared."""
    from huggingface_hub import snapshot_download

    def download(model: str, rev: str, dest: Path,
                 allow_patterns: List[str]) -> None:
        snapshot_download(repo_id=model, revision=rev, local_dir=dest,
                          allow_patterns=allow_patterns)

    status = prepare(Path(VOLUME_MOUNT), model_id, revision, download,
                     datetime.now(timezone.utc).isoformat())
    volume.commit()
    return status


# ---------------------------------------------------------------------------
# Tranche 2: vLLM on a GPU
# ---------------------------------------------------------------------------

# A batch of 64 takes a minute or two at estimated speeds; 30 minutes
# means something is wrong.
BATCH_TIMEOUT_S = 30 * 60

cache_volume = modal.Volume.from_name(
    CACHE_VOLUME_NAME, create_if_missing=True)
CACHE_ENV: Dict[str, Optional[str]] = dict(
    cache_env(CACHE_MOUNT, VLLM_VERSION))

vllm_image = (
    modal.Image.from_registry(VLLM_BASE_IMAGE, add_python='3.12')
    .entrypoint([])
    .uv_pip_install(f'vllm=={VLLM_VERSION}')
    .add_local_python_source('treatments_to_structured')
)


@app.cls(
    image=vllm_image,
    gpu='H100',           # replaced per run by run_translator
    volumes={VOLUME_MOUNT: volume, CACHE_MOUNT: cache_volume},
    env=CACHE_ENV,
    cpu=CONTAINER_CORES,
    memory=CONTAINER_MEMORY_GIB * 1024,
    timeout=BATCH_TIMEOUT_S,
    startup_timeout=20 * 60,
    scaledown_window=SCALEDOWN_WINDOW_S,
    retries=1,            # one retry, e.g. after preemption
)
class JsonTranslator:
    """Mistral-7B under vLLM with JSON-schema-guided, greedy decoding."""

    @modal.enter()
    def load(self) -> None:
        from vllm import LLM

        root = Path(VOLUME_MOUNT)
        if not is_prepared(root, MODEL_ID, MODEL_REVISION):
            raise RuntimeError('model weights are not on the Volume; run '
                               'bin/prepare_modal_models first')
        self.llm = LLM(model=str(model_dir(root, MODEL_ID, MODEL_REVISION)),
                       dtype=DTYPE, max_model_len=MAX_MODEL_LEN,
                       gpu_memory_utilization=0.90, seed=0)
        # Keep what this start-up compiled for the next container.
        cache_volume.commit()
        self.grammars: Dict[str, str] = {}

    @modal.method()
    def translate(self, batch: List[Dict[str, Any]], schema: Dict[str, Any],
                  max_tokens: int) -> List[Dict[str, Any]]:
        """One raw result per ``{_id, prompt}`` in ``batch``, in order.

        Greedy, with the JSON grammar's whitespace runs capped
        (modal_inference.MAX_WHITESPACE_CNT); see json_grammar.
        """
        import json

        import xgrammar
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        key = json.dumps(schema, sort_keys=True)
        if key not in self.grammars:
            self.grammars[key] = json_grammar(schema, xgrammar)
        params = SamplingParams(
            **sampling_options(max_tokens),
            structured_outputs=StructuredOutputsParams(
                grammar=self.grammars[key]))
        outputs = self.llm.generate([b['prompt'] for b in batch], params,
                                    use_tqdm=False)
        return [raw_from_vllm(b['_id'], output)
                for b, output in zip(batch, outputs)]


@contextmanager
def run_translator(
    gpu: str, gpus: int,
) -> Iterator[Callable[[List[Dict[str, Any]], Dict[str, Any], int],
                       List[Dict[str, Any]]]]:
    """Run the app for the block; yield ``call(batch, schema, max_tokens)``.

    At most ``gpus`` containers of ``gpu``.  Leaving the block stops the
    app, so nothing bills afterwards beyond Modal's scale-down window.
    """
    translator_cls = JsonTranslator.with_options(  # type: ignore[attr-defined]
        gpu=gpu, max_containers=gpus)
    with modal.enable_output(), app.run():
        translator = translator_cls()

        def call(batch: List[Dict[str, Any]], schema: Dict[str, Any],
                 max_tokens: int) -> List[Dict[str, Any]]:
            return list(translator.translate.remote(batch, schema,
                                                    max_tokens))

        yield call
