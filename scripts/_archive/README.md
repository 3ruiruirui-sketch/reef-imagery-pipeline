# scripts/_archive

Superseded scripts kept for reference rather than deleted. Nothing here is
imported or referenced by the active codebase. Safe to remove permanently if the
iteration history is no longer useful.

| File | Superseded by | Why archived |
|---|---|---|
| `eval_scene_quality_mini_v2.py` | `scripts/eval_scene_quality_mini_v4.py` | Early iteration of the S2 scene-quality ranker; 0 references |
| `eval_scene_quality_mini_v3.py` | `scripts/eval_scene_quality_mini_v4.py` | Added CRS-aware window; `v4` adds an MGRS tile filter on top; 0 references |

The active versions remain in `scripts/`: `eval_scene_quality_mini.py` (the
canonical base, referenced across the repo) and `eval_scene_quality_mini_v4.py`
(latest: CRS-aware + MGRS tile filter).
