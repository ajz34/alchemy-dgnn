import pandas as pd


origs = [pd.read_csv("targets_valid_{:02d}.csv".format(i)) for i in (0, 1, 2, 3, 4)]
ensemble = (origs[0] + origs[1] + origs[2] + origs[3]) / 4
ensemble["gdb_idx"] = origs[0]["gdb_idx"]
ensemble.set_index("gdb_idx")
ensemble.to_csv("answer.csv", index=False)
