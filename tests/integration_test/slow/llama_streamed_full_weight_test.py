# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from tests.process_test_utils import run_owned_process

REPO = Path(__file__).resolve().parents[3]
JOB = REPO / "examples/advanced/llm_hf/streamed_full_weight/job.py"


def _fixture_model(directory, torch, transformers):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = "[UNK] <s> </s> Instruction Name a fruit Response An apple is vegetable A carrot : .".split()
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(vocab)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    config = transformers.LlamaConfig(
        vocab_size=1050,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        transformers.LlamaForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(directory)
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", bos_token="<s>", eos_token="</s>"
    ).save_pretrained(directory)


def _run_job(*args):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    env.pop("NVFLARE_DENSE_DISK_TASK_TENSORS", None)
    result = run_owned_process([sys.executable, str(JOB), *map(str, args)], cwd=REPO, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.usefixtures("single_threaded_torch")
def test_three_round_streamed_checkpoint_matches_client_mean(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.0")
    from safetensors.torch import load_file

    spec = importlib.util.spec_from_file_location("llama_streamed_full_weight_job", JOB)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)

    base = tmp_path / "base"
    prepared = tmp_path / "prepared"
    sim = tmp_path / "sim"
    restored = tmp_path / "restored"
    _fixture_model(base, torch, transformers)
    _run_job("prepare", base, "--workspace", prepared)
    index = prepared / "model.safetensors.index.json"
    output = _run_job("simulate", base, "--index", index, "--workspace", sim, "--rounds", 3)
    assert output.count("Aggregated 2/2 results") == 3, output
    (checkpoint,) = sim.rglob("FL_global_model.safetensors")

    weight_map = json.loads(index.read_text())["weight_map"]
    assert job.tile_key(job.HEAD, 1024) in weight_map
    expected = {key: load_file(prepared / filename)[key] for key, filename in weight_map.items()}
    config = transformers.AutoConfig.from_pretrained(base, local_files_only=True)
    config._attn_implementation = "eager"
    tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
    for round_number in range(3):
        updates = []
        for site, prompt in job.SAMPLES.items():
            step_output = tmp_path / f"{site}-{round_number}"
            step_output.mkdir()
            owner = job._TempDirRef(str(step_output))
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).input_ids
            _, refs = job.DenseStep(config, expected, owner, "cpu").run(tokens)
            updates.append({key: ref.materialize() for key, ref in refs.items()})
        expected = {key: (updates[0][key] + updates[1][key]) / 2 for key in expected}

    tiled = load_file(checkpoint)
    assert tiled.keys() == expected.keys()
    for key, mean in expected.items():
        torch.testing.assert_close(tiled[key], mean, atol=0, rtol=0, msg=key)

    _run_job("restore", base, "--checkpoint", checkpoint, "--workspace", restored)
    original = load_file(base / "model.safetensors")
    weights = load_file(restored / "model.safetensors")
    assert weights.keys() == original.keys()
    for key, tensor in weights.items():
        parts = [tiled[tile] for tile in sorted(tiled) if tile.startswith(key + ".")]
        torch.testing.assert_close(tensor, torch.cat(parts) if parts else tiled[key], atol=0, rtol=0, msg=key)
    for block in range(config.num_hidden_layers):
        assert any(
            name.startswith(f"model.layers.{block}.") and not torch.equal(tensor, original[name])
            for name, tensor in weights.items()
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(restored, local_files_only=True, dtype=torch.bfloat16)
    restored_tokenizer = transformers.AutoTokenizer.from_pretrained(restored, local_files_only=True)
    with torch.no_grad():
        logits = model(**restored_tokenizer("Name a fruit.", return_tensors="pt")).logits
    assert logits.shape[-1] == config.vocab_size
    assert torch.isfinite(logits).all()
