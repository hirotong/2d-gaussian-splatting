from glob import glob
import numpy as np
import os
import json
import csv

# root = "output/original/TanksAndTemple"
# root = "output/ablation/TanksAndTemple_residual"
# root = "output/ablation/GlossySynthetic_residual"
# root = "output/original/GlossySynthetic"
root = "output/"

DEFAULT_MESH_DICT = {
    "completeness": -1,
    "accuracy": -1,
    "chamfer-L1": -1,
    "f-score": -1,
    "f-score-15": -1,
    "f-score-20": -1,
}

f_names = sorted(glob(os.path.join(root, "*")))
summary = {}
for f_name in f_names:
    results_path = os.path.join(f_name, "results.json")
    if not os.path.exists(results_path):
        continue
    method = None
    with open(results_path) as json_file:
        contents = json.load(json_file)
        # method = sorted(contents.keys(), key=lambda method : int(method.split('_')[-1]))[-1]
        method = sorted(contents.keys(), key=lambda method: float(contents[method]["PSNR"]))[-1]
        try:
            summary["Method"].append(os.path.basename(f_name))
            summary["PSNR"].append(contents[method]["PSNR"])
            summary["SSIM"].append(contents[method]["SSIM"])
            summary["LPIPS"].append(contents[method]["LPIPS"])
        except:
            summary["Method"] = [os.path.basename(f_name)]
            summary["PSNR"] = [contents[method]["PSNR"]]
            summary["SSIM"] = [contents[method]["SSIM"]]
            summary["LPIPS"] = [contents[method]["LPIPS"]]

    mesh_results_path = os.path.join(f_name, "results_mesh.json")
    if not os.path.exists(mesh_results_path):
        mesh_metric_dict = DEFAULT_MESH_DICT
    with open(mesh_results_path) as json_file:
        mesh_metric_dict = json.load(json_file)[method]
    for key, value in mesh_metric_dict.items():
        summary[key] = summary.get(key, []) + [value]

summary["Method"].append("Avg.")
summary["PSNR"].append(np.mean(summary["PSNR"]))
summary["SSIM"].append(np.mean(summary["SSIM"]))
summary["LPIPS"].append(np.mean(summary["LPIPS"]))
for k in DEFAULT_MESH_DICT.keys():
    summary[k].append(np.mean(summary[k]))

with open(os.path.join(root, "summary.csv"), "w") as file_obj:
    writer_obj = csv.writer(file_obj)
    writer_obj.writerow(["Method"] + summary["Method"])
    writer_obj.writerow(["PSNR"] + summary["PSNR"])
    writer_obj.writerow(["SSIM"] + summary["SSIM"])
    writer_obj.writerow(["LPIPS"] + summary["LPIPS"])
    for k in DEFAULT_MESH_DICT.keys():
        writer_obj.writerow([k] + summary[k])
