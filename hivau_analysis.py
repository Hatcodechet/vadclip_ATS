import json
import argparse
from pathlib import Path

def load_json(json_path: str) -> dict:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_buggy_videos(data: dict) -> list[str]:
    """
    Bug definition:
    - label != []
    - len(events_summary) == 1
    - events_summary[0] contains substring "no anomaly" (case-insensitive)
    """
    buggy = []
    for video_name, info in data.items():
        labels = info.get("label", [])
        events_summary = info.get("events_summary", [])

        if labels != [] and len(events_summary) == 1:
            s = str(events_summary[0]).lower()
            if "no anomaly" in s:
                buggy.append(video_name)
    return buggy

def find_videos_having_normal_events(data: dict) -> list[str]:
    """
    Bug definition:
    - label != []
    - len(events_summary) > 1
    - events_summary[i] contains substring "no anomaly" (case-insensitive)
    """
    videos = []
    for video_name, info in data.items():
        labels = info.get("label", [])
        events_summary = info.get("events_summary", [])

        if labels != [] and len(events_summary) > 1:
            for s in events_summary:
                if "no anomaly" in s.lower():
                    videos.append(video_name)
                    break
    return videos
def main():
    data = load_json("HIVAU-70k/raw_annotations/ucf_database_train.json")
    buggy = find_buggy_videos(data)
    contain = find_videos_having_normal_events(data)

    print(f"Error videos count: {len(buggy)}")
    print(f"Contain normal events videos count: {len(contain)}")
    # out_path = Path(args.out)
    out_path = Path("debugs/ucf_database_train/buggy_videos.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(buggy), encoding="utf-8")
    print(f"\nSaved to: {out_path.resolve()}")

    out_path = Path("debugs/ucf_database_train/contain_normal_events_videos.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(contain), encoding="utf-8")
    print(f"\nSaved to: {out_path.resolve()}")
    


if __name__ == "__main__":
    main()