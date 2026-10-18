import argparse
import time
import cv2
from src.capture import VideoSource
from src.pose import PoseEstimator, draw_skeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or a path to a video file")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--skeleton-only", action="store_true",
                        help="hide the camera image (privacy mode)")
    args = parser.parse_args()

    cam = VideoSource(args.source, loop=args.loop)
    pose = PoseEstimator()
    time.sleep(0.5)
    prev = time.time()
    fps = 0.0

    while True:
        frame = cam.read()
        if frame is None:
            if cam.running:
                continue
            break

        landmarks = pose.process(frame)

        if args.skeleton_only:
            frame[:] = 0
        if landmarks is not None:
            draw_skeleton(frame, landmarks)
        else:
            cv2.putText(frame, "No person detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("SafeStep", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pose.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()