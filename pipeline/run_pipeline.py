#!/usr/bin/env python3
"""GeoAgentBench Pipeline v3 — orchestrator script.

Usage:
    python run_pipeline.py                        # run all modules 1-6 (Google Maps source)
    python run_pipeline.py --source gaea          # use GAEA-Train dataset as image source
    python run_pipeline.py --start-from 3         # resume from module 3
    python run_pipeline.py --config path.yaml     # custom config
    python run_pipeline.py --seed-limit 2         # limit seeds for testing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add pipeline root to path so shared/modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.config import PipelineConfig


IMAGE_SOURCES = {
    "google":  (1, "Image Source (Google Maps)", "modules.module_1_image_source"),
    "gaea":    (1, "Image Source (GAEA-Train)",  "modules.module_1_gaea_source"),
}

MODULES_2_TO_6 = [
    (2, "Vision Parser",    "modules.module_2_vision_parser"),
    (3, "Task Builder",     "modules.module_3_task_builder"),
    (4, "Quality Gate",     "modules.module_4_quality_gate"),
    (5, "Query Writer",     "modules.module_5_query_writer"),
    (6, "Review Exporter",  "modules.module_6_review_exporter"),
]


def run(start_from: int = 1, config_path: str | None = None,
        seed_limit: int | None = None, source: str = "google") -> None:
    config = PipelineConfig(config_path) if config_path else PipelineConfig()

    module_1 = IMAGE_SOURCES[source]
    modules = [module_1] + MODULES_2_TO_6

    print("=" * 80)
    print("GeoAgentBench Pipeline v3")
    print(f"  Image source:  {source}")
    print(f"  Vision model:  {config.model_vision}")
    print(f"  Text model:    {config.model_text}")
    print(f"  Strong model:  {config.model_text_strong}")
    print(f"  Start from:    module {start_from}")
    if seed_limit:
        print(f"  Seed limit:    {seed_limit}")
    print("=" * 80)

    for num, name, module_path in modules:
        if num < start_from:
            print(f"\n[Skip] Module {num}: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"[Running] Module {num}: {name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            mod = __import__(module_path, fromlist=["main"])
            if num == 1 and seed_limit is not None:
                mod.main(config=config, seed_limit=seed_limit)
            else:
                mod.main(config=config)
        except Exception as e:
            print(f"\n[FAILED] Module {num}: {name}")
            print(f"  Error: {e}")
            print(f"  Fix the issue and re-run with: --start-from {num}")
            raise

        elapsed = time.time() - t0
        print(f"\n[Done] Module {num}: {name} ({elapsed:.1f}s)")

    print(f"\n{'='*80}")
    print("Pipeline v3 complete!")
    print(f"{'='*80}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoAgentBench Pipeline v3")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from module N (1-6)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--seed-limit", type=int, default=None,
                        help="Limit number of seeds to process (for testing)")
    parser.add_argument("--source", type=str, default="google",
                        choices=["google", "gaea"],
                        help="Image source: 'google' (Street View) or 'gaea' (GAEA-Train)")
    args = parser.parse_args()

    run(start_from=args.start_from, config_path=args.config,
        seed_limit=args.seed_limit, source=args.source)


if __name__ == "__main__":
    main()
