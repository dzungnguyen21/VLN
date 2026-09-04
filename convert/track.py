from libs import *
from config import *
from bag import TRAJ_FILENAME, Bag
from geometry import path_length, resample_indices

import cuvslam
import cv2


MIN_PATH_UNITS = 1e-3
MIN_POSES = 4


def build_mono_tracker(camera_info):
    intrinsics = camera_info["intrinsics"]
    camera = cuvslam.Camera()
    camera.size = (int(camera_info["width"]), int(camera_info["height"]))
    camera.principal = [float(intrinsics[0, 2]), float(intrinsics[1, 2])]
    camera.focal = [float(intrinsics[0, 0]), float(intrinsics[1, 1])]

    return cuvslam.Tracker(
        cuvslam.Rig([camera]),
        cuvslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            rectified_stereo_camera=False,
            odometry_mode=cuvslam.Tracker.OdometryMode.Mono,
        ),
    )


def write_trajectory(out_path, poses, tracked_fps):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        file.write("# timestamp_ns x y z qx qy qz qw (camera optical frame, unknown scale)\n")
        file.write(f"# tracked at {tracked_fps:.2f} fps\n")
        for timestamp_ns, *pose_values in poses:
            values = " ".join(f"{value:.9g}" for value in pose_values)
            file.write(f"{int(timestamp_ns)} {values}\n")


def track_bag(bag_dir, out_dir):
    with Bag.open(bag_dir) as bag:
        camera_info = bag.camera_info()

        if not camera_info["rectified"]:
            raise ValueError("images are not rectified (D != 0)")

        frames = list(bag.iter_jpegs())
        if not frames:
            raise ValueError("the image topic is empty")
        if TRACK_FPS:
            timestamps_ns = np.array([timestamp_ns for timestamp_ns, _ in frames], np.int64)
            kept = resample_indices(timestamps_ns, TRACK_FPS, skip_if_already_slower=True)
            frames = [frames[index] for index in kept]

        tracker = build_mono_tracker(camera_info)

        poses, n_failures = [], 0
        for timestamp_ns, jpeg in frames:
            bgr_image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr_image is None:
                n_failures += 1
                continue

            rgb_image = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
            estimate, _ = tracker.track(timestamp_ns, [rgb_image])
            if estimate.world_from_rig is None:
                n_failures += 1
                continue

            pose = estimate.world_from_rig.pose
            poses.append((timestamp_ns,) + tuple(pose.translation) + tuple(pose.rotation))

    if len(poses) < MIN_POSES:
        raise ValueError(f"only {len(poses)} pose(s) tracked, an episode needs {MIN_POSES}")

    traveled = path_length(np.array(poses)[:, 1:4])
    if traveled <= MIN_PATH_UNITS:
        raise ValueError(f"the trajectory never leaves the origin (path length {traveled:.2e}) — mono VO did not initialise; no motion, or no texture to track")

    span_seconds = (frames[-1][0] - frames[0][0]) / 1e9
    tracked_fps = len(frames) / span_seconds if span_seconds > 0 else 0.0

    out_path = Path(out_dir) / TRAJ_FILENAME
    write_trajectory(out_path, poses, tracked_fps)

    return {
        "path": out_path,
        "n_frames": len(frames),
        "n_poses": len(poses),
        "n_failures": n_failures,
        "fps": tracked_fps,
        "path_length": traveled,
    }
