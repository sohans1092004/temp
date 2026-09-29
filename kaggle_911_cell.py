import glob, os, shutil, subprocess, zlib
import pandas as pd

CODE = glob.glob("/kaggle/input/**/udk_colab_gpu/evaluate_real_world.py", recursive=True)[0].rsplit("/", 1)[0]
V1 = glob.glob("/kaggle/input/**/mdeberta/**/config.json", recursive=True)[0].rsplit("/", 1)[0]
AUDIO = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
calls = [p for p in glob.glob("/kaggle/input/**/*", recursive=True)
         if p.lower().endswith(AUDIO) and "911" in p.lower() and "6_sec" not in p.lower() and "first" not in p.lower()]
if "Files alerting" not in open(CODE + "/evaluate_real_world.py").read():
    raise SystemExit("old code: re-upload the new udk_colab_gpu.zip")
print("code:", CODE, "\nv1:", V1, "\ncalls found:", len(calls))

# false-alarm calls (per the dataset's own metadata) are kept apart from real emergencies
false_alarm = set()
for m in glob.glob("/kaggle/input/**/*911*/**/*.csv", recursive=True) + glob.glob("/kaggle/input/**/911*.csv", recursive=True):
    df = pd.read_csv(m)
    print("metadata:", m, list(df.columns))
    fa_col = next((c for c in df.columns if "false" in c.lower()), None)
    name_col = next((c for c in df.columns if df[c].astype(str).str.contains(r"\.(mp3|wav)", case=False).any()), None)
    if fa_col and name_col:
        false_alarm |= {os.path.basename(str(n)) for n in df.loc[df[fa_col] == 1, name_col]}

# DEV / SEALED split by file-name hash, decided before anyone looks: only DEV is run now
shutil.copytree(CODE, "/kaggle/working/udk_colab_gpu", dirs_exist_ok=True)
os.chdir("/kaggle/working/udk_colab_gpu")
for split in ("dev", "sealed"):
    shutil.rmtree(f"/kaggle/working/911_{split}", ignore_errors=True)
for p in calls:
    split = "sealed" if zlib.crc32(os.path.basename(p).encode()) % 2 else "dev"
    cat = "false_alarm" if os.path.basename(p) in false_alarm else "emergency"
    os.makedirs(f"/kaggle/working/911_{split}/{cat}", exist_ok=True)
    shutil.copy(p, f"/kaggle/working/911_{split}/{cat}/")
for split in ("dev", "sealed"):
    print(split, {c: len(os.listdir(f"/kaggle/working/911_{split}/{c}")) for c in os.listdir(f"/kaggle/working/911_{split}")})

def sh(cmd, env=None):
    p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    open("/kaggle/working/911.log", "a").write(p.stdout + p.stderr)
    return p.stdout

sh("pip install -q -r requirements_colab.txt && pip uninstall -y onnxruntime onnxruntime-gpu && rm -rf /usr/local/lib/python3.*/dist-packages/onnxruntime && pip install --no-deps 'onnxruntime-gpu>=1.22,<1.27,!=1.24.1,!=1.25.*,!=1.26.0'")
env = {**os.environ, "UDK_ENABLE_KWS": "0", "UDK_INTENT_GATE": V1, "UDK_INTENT_THRESHOLD": "0.02"}  # = main, live
sh("python evaluate_real_world.py run --root /kaggle/working/911_dev --max-min 10", env)
print(sh("python evaluate_real_world.py report --root /kaggle/working/911_dev", env))
