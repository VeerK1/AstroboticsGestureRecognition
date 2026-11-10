
import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "yolo11n-pose.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5

# COCO body-keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = YOLO(MODEL_PATH)

def distance(point_a, point_b):
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(point_a - point_b)


def classify_gesture(keypoints, confidences):
    """
    Classify a person's pose into one of 5 distinct long-range gestures:
      1. Hands Raised
      2. One Hand Raised (Either left or right)
      3. T-Pose
      4. Hands on Hips
      5. Crouch
    Returns the gesture name or 'No Gesture'.
    """

    # We require the upper body and hips for most gestures.
    # Knees are checked dynamically later so partial occlusion doesn't break everything.
    required = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP
    ]

    # Require visible landmarks before classifying.
    if any(confidences[i] < CONFIDENCE_THRESHOLD for i in required):
        return "No Gesture"

    ls = keypoints[LEFT_SHOULDER]
    rs = keypoints[RIGHT_SHOULDER]
    le = keypoints[LEFT_ELBOW]
    re = keypoints[RIGHT_ELBOW]
    lw = keypoints[LEFT_WRIST]
    rw = keypoints[RIGHT_WRIST]
    lh = keypoints[LEFT_HIP]
    rh = keypoints[RIGHT_HIP]
    rk = keypoints[RIGHT_KNEE]
    lk = keypoints[LEFT_KNEE]
    ra = keypoints[RIGHT_ANKLE]
    la = keypoints[LEFT_ANKLE]

    shoulder_width = distance(ls, rs)
    if shoulder_width < 4:
        return "No Gesture"

    # Image coordinates: smaller y means higher on screen.
    shoulder_y = (ls[1] + rs[1]) / 2.0
    hip_y = (lh[1] + rh[1]) / 2.0
    torso_height = hip_y - shoulder_y

    ankle_distance = distance(la, ra)

    if torso_height < 6:
        return "No Gesture"

    # --------------------------------------------------
    # 1. CROUCH
    # --------------------------------------------------
    knees_confident = (
        confidences[LEFT_KNEE] > CONFIDENCE_THRESHOLD 
        and confidences[RIGHT_KNEE] > CONFIDENCE_THRESHOLD
    )
    
    if knees_confident:
        lk = keypoints[LEFT_KNEE]
        rk = keypoints[RIGHT_KNEE]
        knee_y = (lk[1] + rk[1]) / 2.0
        
        # When standing straight, knee_y is ~1 torso_height below hip_y.
        # When crouching, the femur is horizontal, compressing the vertical gap.
        if (knee_y - hip_y) < 0.40 * torso_height:
            return "Crouch"

    # --------------------------------------------------
    # Camera Direction & Anatomical Side Resolution
    # --------------------------------------------------
    outward_dir = 1.0 if ls[0] >= rs[0] else -1.0

    # Positive values mean the joint is extended outward away from the torso
    le_outward = (le[0] - ls[0]) * outward_dir
    lw_outward = (lw[0] - ls[0]) * outward_dir
    re_outward = (rs[0] - re[0]) * outward_dir
    rw_outward = (rs[0] - rw[0]) * outward_dir

    # --------------------------------------------------
    # Vertical Zones (Normalized by torso_height)
    # --------------------------------------------------
    raised_margin = 0.20 * torso_height
    left_wrist_high = lw[1] < (ls[1] - raised_margin)
    right_wrist_high = rw[1] < (rs[1] - raised_margin)

    # Horizontal arm up thresholds (Wrist must be directly above shoulder)
    aligned_margin = 0.30 * torso_height
    left_wrist_aligned = (ls[0] + aligned_margin) > le[0] > (ls[0] - aligned_margin)
    right_wrist_aligned = (rs[0] + aligned_margin) > re[0] > (rs[0] - aligned_margin)

    # Neutral "arm down" threshold: wrist is well below chest level
    arm_down_y = shoulder_y + 0.55 * torso_height
    left_wrist_down = lw[1] > arm_down_y
    right_wrist_down = rw[1] > arm_down_y

    # --------------------------------------------------
    # 2. HANDS RAISED (Both hands up)
    # --------------------------------------------------
    if left_wrist_high and left_wrist_aligned and right_wrist_high and right_wrist_aligned:
        return "Hands Raised"

    # --------------------------------------------------
    # 3. ONE HAND RAISED (Either hand up, the other down)
    # --------------------------------------------------
    # if (left_wrist_high and right_wrist_down) or (right_wrist_high and left_wrist_down):
    #     return "One Hand Raised"

    # --------------------------------------------------
    # 3. STAR (Hands up diagonally, legs out diagonally)
    # --------------------------------------------------
    t_vertical_tol = 0.60 * torso_height #0.35 orgiinal

    angleOfLeftArm = np.arctan2(ls[1] - le[1], ls[0] - le[0]) * 180 / np.pi
    angleOfRightArm = np.arctan2(rs[1] - re[1], rs[0] - re[0]) * 180 / np.pi
    arms_at_45 = (
        140 < angleOfLeftArm < 180
        and 20 < angleOfRightArm < 60
    )

    left_arm_straight_out = (
        le_outward > 0.15 * shoulder_width
        and lw_outward > le_outward
        and lw_outward > 0.65 * shoulder_width
    )
    right_arm_straight_out = (
        re_outward > 0.15 * shoulder_width
        and rw_outward > re_outward
        and rw_outward > 0.65 * shoulder_width
    )

    leg_tolerance = shoulder_width * 0.55

    if ankle_distance > shoulder_width + leg_tolerance and arms_at_45:
        return "Star"

    # --------------------------------------------------
    # 4. T-POSE (Both arms extended straight out at shoulder level)
    # --------------------------------------------------
    t_vertical_tol = 0.25 * torso_height

    arms_at_shoulder_height = (
        abs(le[1] - ls[1]) < t_vertical_tol
        and abs(lw[1] - ls[1]) < t_vertical_tol
        and abs(re[1] - rs[1]) < t_vertical_tol
        and abs(rw[1] - rs[1]) < t_vertical_tol
    )

    left_arm_straight_out = (
        le_outward > 0.15 * shoulder_width
        and lw_outward > le_outward
        and lw_outward > 0.65 * shoulder_width
    )
    right_arm_straight_out = (
        re_outward > 0.15 * shoulder_width
        and rw_outward > re_outward
        and rw_outward > 0.65 * shoulder_width
    )

    if arms_at_shoulder_height and left_arm_straight_out and right_arm_straight_out:
        return "T-Pose"


    # --------------------------------------------------
    # 5. FLEX (Both arms Flexing)
    # --------------------------------------------------
    t_vertical_tol = 0.20 * torso_height

    arms_at_shoulder_height = (
        abs(le[1] - ls[1]) < t_vertical_tol
        and abs(re[1] - rs[1]) < t_vertical_tol
    )

    align_wrists_and_elbows = (
        abs(le[0] - lw[0]) < t_vertical_tol
        and abs(re[0] - rw[0]) < t_vertical_tol
    )

    if arms_at_shoulder_height and align_wrists_and_elbows:
        return "Flexing"

    # --------------------------------------------------
    # 5. HANDS ON HIPS (Diamond silhouette)
    # --------------------------------------------------
    # elbows_flared_out = (
    #     le_outward > 0.20 * shoulder_width
    #     and re_outward > 0.20 * shoulder_width
    # )

    # wrists_tucked_in = (
    #     lw_outward < le_outward - 0.10 * shoulder_width
    #     and rw_outward < re_outward - 0.10 * shoulder_width
    # )

    # elbows_mid_torso = (
    #     shoulder_y < le[1] < hip_y + 0.10 * torso_height
    #     and shoulder_y < re[1] < hip_y + 0.10 * torso_height
    # )

    # wrists_near_hips_y = (
    #     abs(lw[1] - hip_y) < 0.35 * torso_height
    #     and abs(rw[1] - hip_y) < 0.35 * torso_height
    # )

    # if elbows_flared_out and wrists_tucked_in and elbows_mid_torso and wrists_near_hips_y:
    #     return "Hands on Hips"

    return "No Gesture"


# --------------------------------------------------
# Main webcam loop
# --------------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check CAMERA_INDEX."
    )

print("Gesture recognition running.")
print("Press 'q' to quit.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        #crop
        height,width,_ = frame.shape
        half_width = width // 2
        frame = frame[:, :half_width]

        img_h, img_w, _ = frame.shape
        
        # Estimate body keypoints for every detected person.
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        # Draw YOLO's pose skeletons and person detections.
        annotated_frame = result.plot()

        if result.keypoints is not None:
            all_keypoints = result.keypoints.xy.cpu().numpy()
            all_confidences = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )

            boxes = (
                result.boxes.xyxy.cpu().numpy()
                if result.boxes is not None
                else []
            )

            for i, points in enumerate(all_keypoints):
                if all_confidences is None:
                    gesture = "No Gesture"
                else:
                    gesture = classify_gesture(
                        points, all_confidences[i]
                    )

                # Place gesture label above the person.
                if len(boxes) > i:
                    x1, y1, _, _ = boxes[i].astype(int)
                    label_x = max(0, x1)
                    label_y = max(30, y1 - 10)
                else:
                    label_x, label_y = 10, 30 + i * 30

                cv2.putText(
                    annotated_frame,
                    gesture,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        cv2.imshow(
            "YOLO Pose Gesture Recognition",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
