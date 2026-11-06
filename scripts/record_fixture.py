"""Record the real model's raw JSON for all messages -> tests/fixtures/qwen3_4b_raw.json.
Re-run after changing the prompt, schema, model or llama-cpp version."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detention_ai.extract import LlamaExtractor  # noqa: E402
from detention_ai.messages import load_feed  # noqa: E402

ex = LlamaExtractor(ROOT / "models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf")
out = {m.id: json.loads(ex.raw_extract(m)) for m in load_feed(ROOT / "data/messages.txt")}
(ROOT / "tests/fixtures/qwen3_4b_raw.json").write_text(
    "{\n" + ",\n".join(f'  "{k}": ' + json.dumps(v, ensure_ascii=False) for k, v in out.items())
    + "\n}\n", encoding="utf-8")
print("recorded", len(out), "messages")
