import jsonlines
from tqdm import tqdm
import json

with jsonlines.open('dev.jsonl') as reader, jsonlines.open("dev_mcqueen_t.jsonl","w") as writer:
    for row in reader:
        outrow={
            "premises":[[row["obs1"],row["hyp1"]],[row["obs1"],row["hyp2"]]],
            "choices" : [row["obs2"],row["obs2"]],
            "id":row["story_id"],
            "gold_label": 0
        }
        writer.write(outrow)
        