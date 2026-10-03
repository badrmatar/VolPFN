import os
import pandas as pd

DUMP_DIR = "accuracy_dumps"


def dump(tag: str, pred_var, realized_var):
    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, f"{tag}.csv")
    pd.DataFrame({"pred_var": pred_var, "realized_var": realized_var}).to_csv(path, index=False)
    print(f"dumped {path}")
