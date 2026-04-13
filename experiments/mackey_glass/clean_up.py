from pathlib import Path
import pickle
import shutil

input_path = Path(__file__).resolve().parent / "msom_results.pkl"
backup_path = Path(__file__).resolve().parent / "msom_results_backup.pkl"

if not input_path.exists():
    raise FileNotFoundError(f"Could not find file: {input_path}")

shutil.copy2(input_path, backup_path)

with open(input_path, "rb") as f:
    results = pickle.load(f)

results = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("state") is not None}

with open(input_path, "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Backup created: {backup_path}")
print(f"Remaining entries: {len(results)}")
