"""트랜스포머 대안 후보(Folio · Strata · Lattice) 비교 실행기 (세부 조절용). 계획된 실험 전체는 main.py 가 돌린다.

    py compare_candidates.py check                       # 구현 검증 (CPU, float64)
    py compare_candidates.py smoke                       # 모든 모델 × 짧은 학습으로 전체 경로 점검
    py compare_candidates.py run --mode joint --memory persistent --seeds 0,1,2
    py compare_candidates.py run --mode separate --tasks kv,mixed,chain --seeds 0,1,2
    py compare_candidates.py run --models folio_nomem,folio_r1 --mode joint --seeds 0,1,2   # 대조군
    py compare_candidates.py measure                     # 처리량·피크 메모리·지연 (단독 실행)
    py compare_candidates.py report                      # results/candidates/report.md

같은 설정의 완료된 실행은 다시 하지 않고, 중단된 실행은 체크포인트에서 이어 간다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import fields

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAIN_MODELS = "transformer,gru,folio,strata,lattice"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_configs(args):
    from candidates.benchmark import RunConfig
    from candidates.tasks import TRAIN_TASKS
    tasks = args.tasks.split(",") if args.tasks else TRAIN_TASKS
    groups = [tuple(tasks)] if args.mode == "joint" else [(t,) for t in tasks]
    common = {f.name: getattr(args, f.name) for f in fields(RunConfig)
              if hasattr(args, f.name) and f.name not in ("variant", "tasks", "seed", "memory")}
    # 난수별로 모델을 번갈아 배치한다: 병렬 실행 때 메모리를 많이 쓰는 모델(Lattice)이 한꺼번에 몰리지 않게
    return [RunConfig(variant=v, tasks=g, memory=mem, seed=s, **common)
            for mem in args.memory.split(",") for g in groups for s in map(int, args.seeds.split(","))
            for v in args.models.split(",")]


def cmd_run(args):
    from candidates.benchmark import run_many
    configs = make_configs(args)
    log(f"실행 {len(configs)}개 (프로세스 {args.workers}개). 이미 끝난 같은 설정은 재사용합니다.")
    t0 = time.perf_counter()
    res = run_many(configs, args.workers, args.device, log)
    log(f"완료: {len(res)}개, {time.perf_counter() - t0:.0f}초")


def cmd_smoke(args):
    args.steps, args.eval_every, args.n_train, args.n_val, args.n_test, args.batch = 6, 3, 16, 8, 8, 8
    args.out_tag = "smoke"
    from candidates import benchmark
    benchmark.OUT = benchmark.OUT / "smoke"                    # 본 결과와 섞이지 않게
    cmd_run(args)


def cmd_measure(args):
    from candidates.benchmark import OUT, measure
    rows = []
    for v in args.models.split(","):
        log(f"측정: {v}")
        rows.append(measure(v, args.device, batch=args.batch, length=256, chunk=args.chunk))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measure.json").write_text(json.dumps({"rows": rows, "measured_at": time.strftime("%Y-%m-%d %H:%M")},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(f"  {r['variant']:18} 파라미터 {r['params']:7,}  학습 {r['train_sec_per_step'] * 1000:7.1f} ms/걸음 "
              f"({r['train_tokens_per_sec']:8,.0f} 토큰/초)  피크 {r['train_peak_bytes'] / 2**20:6.1f} MiB  "
              f"지연 GPU {r['latency']['cuda']['mean_ms']:.2f} ms · CPU {r['latency']['cpu']['mean_ms']:.2f} ms  상태 {r['state_bytes_batch1']:,} B")


def cmd_check(args):
    sys.exit(subprocess.call([sys.executable, "-m", "candidates.checks", "--device", args.device]))


def cmd_report(args):
    from candidates.report import write_report
    path = write_report()
    log(f"보고서: {path}")


def main():
    ap = argparse.ArgumentParser(description="Folio · Strata · Lattice 비교")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "smoke", "measure", "check", "report"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda" if name != "check" else "cpu")
        p.add_argument("--models", default=MAIN_MODELS)
        if name in ("run", "smoke"):
            p.add_argument("--mode", choices=["joint", "separate"], default="joint")
            p.add_argument("--tasks", default="")
            p.add_argument("--memory", default="persistent", help="window, persistent 또는 둘 다(쉼표)")
            p.add_argument("--seeds", default="0,1,2")
            p.add_argument("--steps", type=int, default=1500)
            p.add_argument("--batch", type=int, default=32)
            p.add_argument("--chunk", type=int, default=64)
            p.add_argument("--lr", type=float, default=2e-3)
            p.add_argument("--n_train", type=int, default=2048)
            p.add_argument("--n_val", type=int, default=128)
            p.add_argument("--n_test", type=int, default=256)
            p.add_argument("--eval_every", type=int, default=250)
            p.add_argument("--workers", type=int, default=1)
        if name == "measure":
            p.add_argument("--batch", type=int, default=32)
            p.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    {"run": cmd_run, "smoke": cmd_smoke, "measure": cmd_measure, "check": cmd_check, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
