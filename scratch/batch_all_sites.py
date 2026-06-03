#!/usr/bin/env python3
"""
batch_all_sites.py — Gera imagens melhoradas para todos os locais de mergulho
Reutiliza generate_enhanced_v2.py como módulo, alterando SITE_KEY e DATE_STR.

Uso: python scratch/batch_all_sites.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "generate_enhanced_v2.py"
VENV_PYTHON = Path(__file__).parent.parent / ".venv/bin/python"

# Configurações: (site_key, best_date)
# Datas selecionadas pelos BVI scores mais altos
JOBS = [
    ("pedra_santa_eulalia", "2025-09-25"),
    ("pedra_do_alto",       "2025-09-25"),
    ("tartaruga",           "2025-09-25"),
]

for site_key, date_str in JOBS:
    print(f"\n{'='*60}")
    print(f"  Gerando: {site_key} @ {date_str}")
    print(f"{'='*60}")

    # Patch temporário nas variáveis do script
    code = SCRIPT.read_text()
    code_patched = code.replace(
        f'SITE_KEY   = "pedra_santa_eulalia"',
        f'SITE_KEY   = "{site_key}"'
    ).replace(
        f'DATE_STR   = "2025-09-25"',
        f'DATE_STR   = "{date_str}"'
    )

    tmp = Path(f"/tmp/_batch_{site_key}.py")
    tmp.write_text(code_patched)

    result = subprocess.run(
        [str(VENV_PYTHON), str(tmp)],
        cwd=str(SCRIPT.parent.parent),
        capture_output=False,
        text=True,
    )
    tmp.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"❌ Erro no {site_key}!")
    else:
        print(f"✅ {site_key} concluído")

print("\n\n🎉 Batch completo!")
