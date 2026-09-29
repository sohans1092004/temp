# UDK — working context (read this first, keep it short, update it when state changes)

Last updated: 2026-09-29

## Where things are
- Live model: `main` @ 96ea038 on AWS (push to main auto-deploys — never push main).
- Current branch: `intent-context` (local only, uncommitted eval/mining files).
- Docs (Claude Docs): "UDK Project Brief" https://claude.ai/code/artifact/04b1b914-f063-436f-bd13-b82e2c9e26f9 (1 page) ·
  "UDK Phase 0: Baseline Specification" https://claude.ai/code/artifact/0f41b822-080a-4861-acc7-6cfee1b33e95 (audit) ·
  "UDK System: What We Did" (full history; results = sections 16, 17).
- `PROJECT_CONTEXT.md` = OLD 2026-09-19 design review (~50k tokens, outdated). Don't read unless asked.
- `README.md` = per-milestone build history. `sealed_runs.md` = sealed test log.

## Live pipeline (api.py -> udk_engine.py)
vad.py (webrtcvad, <=8 s segments) -> stt.py (Whisper small.en + Parakeet TDT 0.6B v2) ->
udk_engine.py matcher (exact/fuzzy/semantic.py MiniLM >=0.65; >=0.60 VERIFY, >=0.85 ALL; repeat/2nd UDK in 15 s -> ALL) ->
stt_confidence_gate.py -> beats_distress_detector.py (vocal >=0.30 -> ALL; scream_trigger.py scream-alone -> VERIFY) ->
intent_gate.py (mDeBERTa HF sohans1092004/udk-intent-mdeberta @6131acc0, veto P<0.02). Config: deploy/.env.example + SSM.

## Eval scripts (run on Kaggle/Colab by the user, never locally with real models)
- evaluate_prepped_data.py (prepped_data/, 78/84 danger, 26/162 controls — UDK path only)
- evaluate_real_world.py (FA/h on long real audio; `--sealed` = totals only; "Files alerting" col for recall sets)
- evaluate_full_system.py (shared run_file_either); package: build_colab_gpu_package.py -> udk_colab_gpu.zip
- kaggle_911_cell.py = 911-calls recall test cell (run as Save Version)

## Baseline numbers (never combine across datasets)
real_recordings mixed 23/30, positives 3/7, negatives 1/23 · dev FA/h conversation 5.3, hard 12.2, ambient 3.9, Telugu 4.2 ·
sealed run 1: W+P 2.2/h, everything live 4.6/h.

## Rejected / parked (don't re-pitch)
WeSep/SepFormer separation · intent v2 context model (kills cries after attacker speech) · LLM-generated data ·
ConVAWG · generic multilingual STT for Indic (Whisper outputs English 95-99%).

## Pending
1. 911 full dev rerun (kaggle_911_cell.py) -> analyse missed calls -> update status doc s17 + Phase 0 doc.
2. Scream FA: needs new dev audio batch from user.
3. Phase 1 (dataset discovery): only when user says.
4. Known bugs (Phase 0, unfixed): Denver placeholder UDK live; retention.sweep/evict_finished never called;
   Whisper load failure -> silent MockSTT; unpinned deps/models; crowd noise = white noise, unfixed seed.

## Rules
Recall first · sealed protocol (split dev/sealed before looking) · general fixes, no per-example rules ·
feature branch only, no Claude co-author · ask before any real-model run.
