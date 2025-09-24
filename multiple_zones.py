
import cv2
import os
import json
import argparse
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict, deque

# =================== Helper Functions ===================

def load_zones(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Map YOLO classes to categories of interest
INTEREST_CLASSES = {
    "person": "human",
    "cat": "animal", "dog": "animal", "horse": "animal", "cow": "animal", "sheep": "animal",
    "elephant": "animal", "bear": "animal", "zebra": "animal", "giraffe": "animal",
    "car": "vehicle", "motorbike": "vehicle", "bus": "vehicle", "truck": "vehicle",
    "bicycle": "vehicle", "train": "vehicle"
}

# Processor Mapping
processors = {
    1:"cpu",
    2:"cuda",
    3:"mps"
}

# =================== Main Function ===================

def motion_detection(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {args.input}")
        return

    # Load YOLO model
    model = YOLO(args.yolo_model)
    try:
        model.to(processors[args.processor])
    except Exception as e:
        print(f"Error setting processor: {e}")
        print("Defaulting to CPU")
        model.to("cpu")


    # Load zones
    zones = load_zones(args.json_file)

    # Background Subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Prepare output
    os.makedirs(args.output, exist_ok=True)
    events = []

    frame_count = 0
    zone_frame_counts = {}   # continuous counts
    active_events = {}       # currently recording videos
    buffer_frames = defaultdict(lambda: deque())  # store frames until threshold breached

    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_interval = 1.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_time = frame_count * frame_interval

        # Keep a copy for drawing overlays
        frame_out = frame.copy()

        # Background subtraction mask
        fg_mask = back_sub.apply(frame)

        # YOLO detections
        results = model(frame, conf=0.4, iou=args.iou, verbose=False)

        # For drawing bounding boxes
        moving_boxes = []

        # Track per-zone motion
        for zone in zones:
            cam_id = zone["camera_id"]
            zone_name = zone["zone_name"]
            poly = zone["zone_perimeter_co_ordinates"]
            threshold = zone["threshold_num_of_frames"]
            key = (cam_id, zone_name)

            motion_detected = False

            # Process detections
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    if cls_name not in INTEREST_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if not point_inside_polygon((cx, cy), poly):
                        continue

                    roi_mean = fg_mask[y1:y2, x1:x2].mean()
                    if roi_mean < 25:
                        continue

                    motion_detected = True

                    moving_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "detected_object": INTEREST_CLASSES[cls_name],
                        "confidence": float(box.conf[0])
                    })

            # Update continuous motion frame count
            if motion_detected:
                zone_frame_counts[key] = zone_frame_counts.get(key, 0) + 1
                buffer_frames[key].append((frame_out.copy(), frame_time))
            else:
                # If motion stops but an active event is ongoing → close it
                if key in active_events:
                    active_events[key]["video_writer"].release()
                    events.append({
                        "camera_id": cam_id,
                        "zone_name": zone_name,
                        "start_time_stamp": str(active_events[key]["start_time"]),
                        "end_time_stamp": str(frame_time),
                        "event_detected": "motion",
                        "video_path": active_events[key]["video_path"]
                    })
                    del active_events[key]

                zone_frame_counts[key] = 0
                buffer_frames[key].clear()

            # If threshold breached and no active event yet → start video
            if zone_frame_counts.get(key, 0) == threshold and key not in active_events:
                # Create video writer
                video_path = os.path.join(args.output, f"{cam_id}_{zone_name}_{frame_count}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

                # Flush buffered frames into video
                start_time_event = buffer_frames[key][0][1] if buffer_frames[key] else frame_time
                for bf, _ in buffer_frames[key]:
                    video_writer.write(bf)

                active_events[key] = {
                    "start_time": start_time_event,
                    "video_writer": video_writer,
                    "video_path": video_path
                }

             # Draw zones
        for zone in zones:
            poly = np.array(zone["zone_perimeter_co_ordinates"], dtype=np.int32)
            cv2.polylines(frame_out, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame_out, zone["zone_name"], tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display continuous frame count
            key = (zone["camera_id"], zone["zone_name"])
            count = zone_frame_counts.get(key, 0)
            cv2.putText(frame_out, f"Frames: {count}", (poly[0][0], poly[0][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw detections
        for box in moving_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = f"{box['detected_object']} {box['confidence']:.2f}"
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            # If active event, keep writing
        if key in active_events and motion_detected:
            active_events[key]["video_writer"].write(frame_out)

       
        # Show live window
        cv2.imshow("AI Inference", frame_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup any still-active events
    for key, event in active_events.items():
        event["video_writer"].release()
        events.append({
            "camera_id": key[0],
            "zone_name": key[1],
            "start_time_stamp": str(event["start_time"]),
            "end_time_stamp": str(frame_count * frame_interval),
            "event_detected": "motion",
            "video_path": event["video_path"]
        })

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"[INFO] Total elapsed time: {elapsed:.2f} sec for {frame_count} frames")

    # Save JSON output
    out_file = os.path.join(args.output, "events.json")
    with open(out_file, "w") as f:
        json.dump(events, f, indent=4)
    print(f"[INFO] Events saved to {out_file}")


# =================== Entry Point ===================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Detection with YOLO + Zones")
    parser.add_argument("--input", required=True, help="RTSP link or video file")
    parser.add_argument("--json_file", required=True, help="Path to zones JSON config")
    parser.add_argument("--yolo_model", required=True, help="YOLO model file path")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--processor", type=int, choices=[1,2,3], default=1, help="1: CPU, 2: CUDA, 3: MPS")
    args = parser.parse_args()

    motion_detection(args)
