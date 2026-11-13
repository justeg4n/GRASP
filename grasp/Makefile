.PHONY: setup lint test data index serve check collect export overnight sft grpo eval
setup:   ; uv sync --extra dev
lint:    ; uv run ruff check . && uv run ruff format --check .
test:    ; uv run pytest
data:    ; uv run python scripts/prepare_data.py
index:   ; uv run python scripts/build_index.py data=$(DATA)
serve:   ; bash scripts/serve_reader.sh $(or $(MODE),fp8)
check:   ; uv run python scripts/check_api.py +check_reader=true
collect: ; uv run python scripts/collect_sft.py data=$(DATA)
export:  ; uv run python scripts/export_sft.py
overnight: ; bash scripts/run_overnight.sh
sft:     ; uv run python scripts/train.py train=sft
grpo:    ; uv run python scripts/train.py train=grpo_g4
eval:    ; uv run python scripts/evaluate.py
