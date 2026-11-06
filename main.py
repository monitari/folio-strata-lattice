"""한 번에 실행: 빠진 실험만 채우고 실험 노트(웹)를 연다.

    py main.py                  # 패키지 확인 → 검증 → 빠진 학습 → 추가 시험 · 지연별 평가 → 자원 측정 → 보고서 → 서버 → 브라우저
    py main.py --check          # 무엇이 준비됐는지만 보여준다
    py main.py --view           # 계산 없이 지금 있는 결과로 화면만 연다
    py main.py --remeasure      # 자원 측정을 다시 한다 (학습 결과는 그대로)
    py main.py --workers 3 --no-browser --port 9000

Windows에서는 start.bat 을 두 번 눌러도 된다. 계획(candidates/plan.py)과 설정이 같은 실행은 다시 하지 않고,
중간에 멈춘 실행은 체크포인트에서 이어 간다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TORCH_GPU_INDEX = "https://download.pytorch.org/whl/cu128"
# 병렬 실행에서 잰 실행 하나의 걸음당 초 (평가 포함). 순환형 후보는 난수 묶음 학습(candidates/fast.py) 기준. 예상 시간 안내용
SEC_PER_STEP = {"transformer": 0.05, "gru": 0.07, "folio": 0.12, "strata": 0.15, "lattice": 0.4}

if hasattr(sys.stdout, "reconfigure"):   # Windows 콘솔에서도 한글이 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. 패키지
def ensure_packages():
    missing = [m for m in ("numpy", "torch") if importlib.util.find_spec(m) is None]
    if not missing:
        return
    has_gpu = shutil.which("nvidia-smi") is not None
    cmds = []
    if "numpy" in missing:
        cmds.append([sys.executable, "-m", "pip", "install", "numpy"])
    if "torch" in missing:
        cmds.append([sys.executable, "-m", "pip", "install", "torch", *(["--index-url", TORCH_GPU_INDEX] if has_gpu else [])])
    print(f"필요한 패키지가 없습니다: {', '.join(missing)}"
          + (" (NVIDIA GPU가 보여서 GPU용 PyTorch, 약 3GB)" if has_gpu and "torch" in missing else ""))
    for c in cmds:
        print("   ", " ".join(c[1:]))
    if input("지금 설치할까요? [y/N] ").strip().lower() != "y":
        sys.exit("설치를 건너뛰었습니다. 위 명령을 직접 실행한 뒤 다시 실행하세요.")
    for c in cmds:
        subprocess.check_call(c)


# ---------------------------------------------------------------- 2. 준비 상태
def status() -> dict:
    import torch
    from candidates import delay, knobs, posthoc
    from candidates.benchmark import OUT
    from candidates.models import VARIANTS
    from candidates.tasks import SUITE
    from candidates.plan import GROUPS, configs

    missing = {g: [c for c in configs(g) if not (OUT / "runs" / c.run_id() / "result.json").exists()] for g in GROUPS}
    chain_runs = [p for p in (OUT / "runs").glob("*/result.json")
                  if "chain" in json.loads(p.read_text(encoding="utf-8"))["tasks"]] if (OUT / "runs").exists() else []
    posthoc_missing = [p for p in chain_runs if not p.with_name("posthoc.json").exists()
                       or json.loads(p.with_name("posthoc.json").read_text(encoding="utf-8")).get("version") != posthoc.POSTHOC_VERSION]
    kvd_runs = [p for p in (OUT / "runs").glob("*/result.json")
                if any(t.startswith("kvd") for t in json.loads(p.read_text(encoding="utf-8"))["tasks"])] if (OUT / "runs").exists() else []
    delay_missing = [p for p in kvd_runs if not p.with_name("delay.json").exists()
                     or json.loads(p.with_name("delay.json").read_text(encoding="utf-8")).get("version") != delay.DELAY_VERSION]
    suite_runs = [p for p in (OUT / "runs").glob("*/result.json")
                  if set(json.loads(p.read_text(encoding="utf-8"))["tasks"]) <= set(SUITE)] if (OUT / "runs").exists() else []
    knob_missing = [p for p in suite_runs if not p.with_name("knobs.json").exists()
                    or json.loads(p.with_name("knobs.json").read_text(encoding="utf-8")).get("version") != knobs.KNOB_VERSION]
    measured = set()
    if (OUT / "measure.json").exists():
        measured = {r["variant"] for r in json.loads((OUT / "measure.json").read_text(encoding="utf-8"))["rows"]}
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "groups": {g: (len(configs(g)) - len(missing[g]), len(configs(g)), GROUPS[g]) for g in GROUPS},
        "missing": [c for g in GROUPS for c in missing[g]],
        "posthoc_missing": len(posthoc_missing) + len(delay_missing) + len(knob_missing),
        "measure_ok": set(VARIANTS) <= measured,
        "checks_ok": (OUT / "checks.json").exists(),
    }


def show(st: dict):
    mark = lambda ok: "준비됨" if ok else "없음"
    print(f"\n  계산 장치          {st['device']}")
    print("  구현 검증          " + mark(st["checks_ok"]))
    for g, (done, total, label) in st["groups"].items():
        print(f"  실험 {g}             {done:>3}/{total:<3} {label}")
    left = st["posthoc_missing"]
    print(f"  추가 시험          {'준비됨' if not left else f'{left}개 남음'}")
    print(f"  자원 측정          {mark(st['measure_ok'])}\n")


def estimate(st: dict, workers: int) -> str:
    sec = sum(c.steps * SEC_PER_STEP.get(c.variant.split("_")[0], 1.0) for c in st["missing"]) / max(1, workers)
    sec += (0 if st["measure_ok"] else 300) + (0 if st["checks_ok"] else 400)
    if st["device"] != "cuda":
        sec *= 10
    return f"약 {sec / 60:.0f}분" if sec >= 60 else "1분 이내"


# ---------------------------------------------------------------- 3. 계산
def fill(st: dict, args):
    from candidates import delay, knobs, posthoc
    from candidates.benchmark import OUT, measure, run_many
    from candidates.checks import run_all
    from candidates.models import VARIANTS
    from candidates.report import write_report

    if not st["checks_ok"]:
        log("구현 검증 (CPU, float64)")
        if run_all("cpu", VARIANTS, True, OUT / "checks.json"):
            log("검증에 실패한 항목이 있습니다. 결과 화면의 '검증' 절에서 확인하세요.")
    if st["missing"]:
        workers = args.workers if st["device"] == "cuda" else 1
        log(f"빠진 학습 {len(st['missing'])}개를 프로세스 {workers}개로 돌립니다.")
        run_many(st["missing"], workers, st["device"], log)
    log("추가 시험 (관계 사슬 단계 수, 키-값 지연 구간별, 새 과제 난이도별)")
    posthoc.run(st["device"], log)
    delay.run(st["device"], log)
    knobs.run(st["device"], log)
    if not st["measure_ok"] or args.remeasure:
        log("자원 측정 (다른 학습이 없는 상태에서 한 모델씩)")
        rows = [measure(v, st["device"], batch=64, length=256, chunk=64) for v in VARIANTS]
        (OUT / "measure.json").write_text(json.dumps({"rows": rows, "measured_at": time.strftime("%Y-%m-%d %H:%M")},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"보고서: {write_report()}")


# ---------------------------------------------------------------- 4. 서버
def already_serving(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/results", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Folio · Strata · Lattice 실험 노트: 한 번에 실행")
    parser.add_argument("--check", action="store_true", help="준비 상태만 보여준다")
    parser.add_argument("--view", action="store_true", help="계산 없이 화면만 연다")
    parser.add_argument("--remeasure", action="store_true", help="자원 측정을 다시 한다")
    parser.add_argument("--workers", type=int, default=5, help="동시에 돌릴 학습 프로세스 수 (GPU)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ensure_packages()
    sys.path.insert(0, str(ROOT))
    st = status()
    show(st)
    if args.check:
        return

    todo = st["missing"] or st["posthoc_missing"] or not st["measure_ok"] or not st["checks_ok"] or args.remeasure
    if todo and not args.view:
        if already_serving(args.port):
            sys.exit(f"포트 {args.port}에서 서버가 돌고 있습니다. 측정이 섞이지 않도록 서버를 끄고 다시 실행하세요 (또는 --view).")
        log(f"빠진 것을 준비합니다. 예상 시간 {estimate(st, args.workers)}. 중간에 멈춰도 끝난 실행은 남고, 다음에 이어 갑니다.")
        fill(st, args)
        log("준비 완료")
        show(status())

    if already_serving(args.port):
        log(f"이미 실행 중인 실험 노트를 엽니다: http://localhost:{args.port}")
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}")
        return
    from server import serve
    serve(args.port, not args.no_browser)


if __name__ == "__main__":
    main()
