import os
import re
from collections import defaultdict
from datetime import datetime

BASE_DIR = "../silm-eval/models/"

TOTAL_STEPS = {
    "en": 523000,
    "de": 506000,
    "zh": 473000,
}

step_pattern = re.compile(r"step_(\d+)")

results = []
language_summary = defaultdict(list)

for language in TOTAL_STEPS.keys():
    lang_path = os.path.join(BASE_DIR, language)
    if not os.path.isdir(lang_path):
        continue

    for model in os.listdir(lang_path):
        model_path = os.path.join(lang_path, model)
        if not os.path.isdir(model_path):
            continue

        max_step = 0

        for item in os.listdir(model_path):
            match = step_pattern.match(item)
            if match:
                step = int(match.group(1))
                max_step = max(max_step, step)
                if max_step == step:
                    latest_ckpt_path = os.path.join(model_path, item)

        total = TOTAL_STEPS.get(language)
        if total is None:
            continue

        percent = max_step / total
        complete = max_step >= total

                # ---- timestamp ----
        mtime = os.path.getmtime(latest_ckpt_path)
        dt = datetime.fromtimestamp(mtime)
        formatted_time = dt.strftime("%Y-%m-%d %H:%M")

        results.append((language, model, max_step, total, percent, complete, formatted_time))
        language_summary[language].append(percent)

# ---- DISPLAY ----

print("\nMODEL STATUS\n")

for language, model, step, total, percent, complete, timestamp in sorted(results):
    bar = "#" * int(percent * 20)
    print(f"{language}/{model:20} {step:6}/{total} "
          f"[{bar:<20}] {percent*100:6.1f}% "
          f"{'DONE' if complete else ''}"
          f"{timestamp if not complete else ''}")

print("\nLANGUAGE SUMMARY\n")

for language, percents in language_summary.items():
    avg = sum(percents) / len(percents)
    done = sum(p == 1.0 for p in percents)
    print(f"{language:5}  avg: {avg*100:6.1f}%   "
          f"completed: {done}/{len(percents)}")

overall_avg = sum(p for _,_,_,_,p,_,_ in results) / len(results)

print("\nOVERALL")
print(f"Average completion: {overall_avg*100:.1f}%")
