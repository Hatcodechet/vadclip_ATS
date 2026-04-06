import os
import json
import pandas as pd

xlsx_path = "/Users/hunghehe2205/Projects/VAD/Capstone/UCF-AR/ucf_crime_train.xlsx"
out_dir = os.path.dirname(xlsx_path)
out_path = os.path.join(out_dir, "ucf_crime_train.json")

df = pd.read_excel(xlsx_path, sheet_name=0)


df = df[["Video Name", "English Text"]].copy()

# Bỏ \n và whitespace ở đầu/cuối
df["Video Name"] = df["Video Name"].astype(str).str.strip()
df["English Text"] = df["English Text"].astype(str).str.strip()

# Thay "nan" thành None
df["Video Name"] = df["Video Name"].replace("nan", None)
df["English Text"] = df["English Text"].replace("nan", None)

data = df.to_dict(orient="records")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Wrote:", out_path)
