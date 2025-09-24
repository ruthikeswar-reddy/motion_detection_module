# Motion Detection with YOLO and Configurable Zones

This script performs motion detection and object classification within user-defined zones in a video stream using a YOLO model. It records events and saves video clips when motion is detected for a specified number of consecutive frames in each zone.

---

## Input Explanation

### 1. Video Input
- **`--input`**: Path to the video file or RTSP stream to be analyzed (e.g., `resize.mp4`).

### 2. Zones Configuration (`zones.json`)
- **`--json_file`**: Path to a JSON file defining the zones to monitor.
- Each zone object in the JSON should have:
  - `camera_id`: Identifier for the camera.
  - `zone_name`: Name of the zone.
  - `zone_perimeter_co_ordinates`: List of `[x, y]` coordinates defining the polygon for the zone.
  - `threshold_num_of_frames`: Number of consecutive frames with motion required to trigger an event.

**Example `zones.json`:**
```json
[
  {
    "camera_id": "cam1",
    "zone_name": "Front Gate",
    "zone_perimeter_co_ordinates": [[1619,1159],[1976,1464],[2479,1029],[2020,872]],
    "threshold_num_of_frames": 10
  },
  {
    "camera_id": "cam1",
    "zone_name": "Parking Lot",
    "zone_perimeter_co_ordinates": [[1691,1192],[1025,1422],[2730,2048],[3075,1509]],
    "threshold_num_of_frames": 30
  }
]
```

### 3. YOLO Model
- **`--yolo_model`**: Path to the YOLO weights file (e.g., `yolo11n.pt`).

### 4. Other Parameters
- **`--iou`**: IOU threshold for YOLO detections (default: 0.5).
- **`--output`**: Directory where event videos and JSON logs will be saved.
- **`--processor`**: Device to run inference on (`1`: CPU, `2`: CUDA, `3`: MPS for Apple Silicon).

---

## Output Explanation

### 1. Event Videos
- For each detected event (motion in a zone for the required number of frames), a video clip is saved in the output directory.
- Filename format: `<camera_id>_<zone_name>_<frame_number>.mp4`

### 2. Event Log (`events.json`)
- A JSON file summarizing all detected events is saved in the output directory.
- Each event contains:
  - `camera_id`: Camera identifier.
  - `zone_name`: Name of the zone.
  - `start_time_stamp`: When the event started.
  - `end_time_stamp`: When the event ended.
  - `event_detected`: Always `"motion"` for this script.
  - `video_path`: Path to the saved event video.

**Example `events.json`:**
```json
[
  {
    "camera_id": "cam1",
    "zone_name": "Front Gate",
    "start_time_stamp": "2024-06-10 12:34:56",
    "end_time_stamp": "2024-06-10 12:35:10",
    "event_detected": "motion",
    "video_path": "./events_out/cam1_Front Gate_1234.mp4"
  }
]
```

---

## Example Command

```sh
python3 multiple_zones.py \
  --input resize.mp4 \
  --json_file ./end_user_details_zones_configurations/zones.json \
  --yolo_model yolo11n.pt \
  --iou 0.45 \
  --output ./events_out \
  --processor 3
```

---

## Notes

- The script displays a live window with zone overlays and detected objects. Press `q` to quit.
- Only objects of interest (e.g., person, car, animal, vehicle) are considered for motion events.
- Make sure the YOLO model and all dependencies are installed and compatible with your processor.


---
## Enhancements needed

- The start time stamp and end time stamp format needed to be changed.
- The video saving needed to be enhanced with the results being correctly written on frames, and the video file name saving format.
- The threshold frames will be changed threshold seconds in future version.
