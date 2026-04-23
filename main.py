import cv2
import argparse
from mediapipe.python.solutions.pose import Pose, PoseLandmark
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (camera index or file path)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="capture.json",
        help="The output filename (json)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to capture"
    )
    return parser.parse_args()


def pose_detection(source, max_frames=None):
    cap = cv2.VideoCapture(source)
    pose = Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_count = 0
    results = []

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break
        results.append(pose.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)))
        frame_count += 1

    capture = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "landmarks": [
            [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
            for res in results if res.pose_landmarks
        ],
    }
    cap.release()
    cv2.destroyAllWindows()
    return capture


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    capture = pose_detection(source, args.max_frames)
    factor = np.array([
        -(5.0 * capture["width"] / capture["height"]), 0.5, -5.0
    ])
    scaled_landmarks = (np.array(capture["landmarks"])[:, :, [0, 2, 1]] - 0.5) * factor
    ref_frame = scaled_landmarks[0]

    with open("armature-lookup.json", "r") as f:
        armature_lookup = json.load(f)

    def build_armature(d):
        result = {}

        for key, v in d.items():
            result[key] = {
                "head": np.mean(
                    np.array([ref_frame[PoseLandmark[i].value] for i in v["head"]]),
                    axis=0
                ).tolist(),
                "tail": np.mean(
                    np.array([ref_frame[PoseLandmark[i].value] for i in v["tail"]]),
                    axis=0
                ).tolist(),
            }

            children = v.get("children")

            if children:
                result[key]["children"] = build_armature(children)

        return result

    armature = build_armature(armature_lookup)

    with open("armature.json", "w") as f:
        json.dump(armature, f, indent=2)

    capture["landmarks"] = scaled_landmarks.tolist()
    with open(args.out, "w") as f:
        json.dump(capture, f, indent=2)


if __name__ == "__main__":
    main()
