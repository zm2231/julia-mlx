from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from tokenizers import Tokenizer


class JuliaTokenizer:
    def __init__(self, directory: str | Path, cache_size: int = 8192):
        root = Path(directory)
        config = json.loads((root / "tokenizer_config.json").read_text())
        self.backend = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.mask_token = config["mask_token"]
        self.mask_token_id, self.cls_token_id, self.sep_token_id, self.pad_token_id = (
            self._token_id(config[key]) for key in ("mask_token", "cls_token", "sep_token", "pad_token"))
        if cache_size < 0:
            raise ValueError("token cache size must be nonnegative")
        self.cache_size = cache_size
        self.cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    def _token_id(self, token: str) -> int:
        value = self.backend.token_to_id(token)
        if value is None:
            raise ValueError(f"tokenizer does not define {token}")
        return value

    def encode(self, text: str) -> list[int]:
        cached = self.cache.get(text)
        if cached is not None:
            self.cache.move_to_end(text)
            return list(cached)
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if self.cache_size:
            self.cache[text] = tuple(ids)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return ids
