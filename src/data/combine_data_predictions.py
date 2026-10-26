import argparse
import json
import os
import pandas as pd
import statistics
import re

def combine_results(data: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    results = {}

    for id in predictions.keys():
        id_int = int(id)
        prediction = predictions[id]
        print(f"Processing {id_int}")
        try:
            instance = data.loc[id_int]
        except KeyError:
            print(f"No entry for {id_int}")
            continue

        instance_dict = instance.to_dict()
        instance_dict["iou"] = prediction["iou"] 
        instance_dict["lrp"] = prediction["lrp"]
        
        results[id_int] = instance_dict

    return pd.DataFrame.from_dict(results, orient="index")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name (e.g., yolo, rf-detr, faster-rcnn).")
    parser.add_argument("features", choices=["metafeatures", "llm-metafeatures"], help="Select which feature set to use.")
    parser.add_argument("--discretize", action="store_true", help="Use discretized detections.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    output_dir = "./data"

    if args.features == "metafeatures":
        data_path = f"./data/{args.features}.csv"
        feature_tag = "metafeatures"
    elif args.features == "llm-metafeatures":
        data_path = "./data/llm_metafeatures.csv"
        feature_tag = "llm-metafeatures"
    else:
        raise ValueError(f"Invalid features choice: {args.features}")

    predictions_file = "detections_disc.json" if args.discretize else "detections.json"
    predictions_path = os.path.join("./results", args.model, predictions_file)

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    data_df = pd.read_csv(data_path, index_col=0)
    with open(predictions_path) as f:
        predictions_dict = json.load(f)

    final_df = combine_results(data_df, predictions_dict)
    print(final_df.head())

    output_path = os.path.join(output_dir, f"{args.model}_{feature_tag}{'_disc' if args.discretize else ''}.csv")
    final_df.to_csv(output_path, index=True)
