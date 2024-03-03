import os
import subprocess
import sys

from tqdm.auto import tqdm

os.environ["LD_LIBRARY_PATH"] = "/opt/conda/lib/"

parameters = {
    "wikipedia_1": {
        "path": "../tmp/Wikipedia_Group_1_bias",
        "labels": ["label", "label_0", "label_1", "label_2"],
    },
    "wikipedia_2": {
        "path": "../tmp/Wikipedia_Group_2_bias",
        "labels": ["label", "label_0", "label_1", "label_2", "label_3"],
    },
}

path_logs = "../tmp/logs/"
for dataset in tqdm(parameters.keys()):
    for label in tqdm(parameters[dataset]["labels"]):
        with open(path_logs + "stdout.log", "a") as fout:
            with open(path_logs + "log.log", "a") as ferr:
                proc = subprocess.call(
                    [
                        "python",
                        "./99_trainer.py",
                        "-l",
                        label,
                        "-d",
                        dataset,
                        "-p",
                        parameters[dataset]["path"],
                    ],
                    stdout=fout,
                    stderr=ferr,
                )
