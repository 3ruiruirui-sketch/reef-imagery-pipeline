#!/usr/bin/env python3
"""
watchdog.py — Dashboard Watchdog
=================================
Mantém o servidor Flask do dashboard sempre a correr.
Reinicia automaticamente se o processo morrer.

Uso:
    python watchdog.py
    # Ctrl+C para parar
"""
import subprocess
import time
import sys
import os
import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [WATCHDOG]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
APP_CMD       = [sys.executable, "app.py"]
RESTART_DELAY = 3   # segundos entre restarts
MAX_RESTARTS  = 20  # segurança: parar após N restarts consecutivos em curto espaço

_proc = None

def _cleanup(sig, frame):
    global _proc
    logging.info("Sinal %s recebido — a terminar servidor...", sig)
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    sys.exit(0)

signal.signal(signal.SIGINT,  _cleanup)
signal.signal(signal.SIGTERM, _cleanup)

def run():
    global _proc
    restarts = 0
    last_restart_time = 0.0

    logging.info("Watchdog iniciado — a gerir %s/app.py", DASHBOARD_DIR)
    logging.info("Ctrl+C para parar")

    while True:
        now = time.time()

        # Reset contador se último restart foi há >60s (servidor estava estável)
        if now - last_restart_time > 60:
            restarts = 0

        if restarts >= MAX_RESTARTS:
            logging.error(
                "Atingido limite de %d restarts consecutivos — a parar watchdog.",
                MAX_RESTARTS
            )
            sys.exit(1)

        logging.info("A iniciar servidor (restart #%d)...", restarts + 1)
        last_restart_time = now

        _proc = subprocess.Popen(
            APP_CMD,
            cwd=DASHBOARD_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        exit_code = _proc.wait()

        if exit_code == 0:
            logging.info("Servidor terminou limpo (exit 0) — a reiniciar em %ds...", RESTART_DELAY)
        else:
            logging.warning(
                "Servidor terminou com código %d — a reiniciar em %ds...",
                exit_code, RESTART_DELAY
            )

        restarts += 1
        time.sleep(RESTART_DELAY)

if __name__ == "__main__":
    run()
