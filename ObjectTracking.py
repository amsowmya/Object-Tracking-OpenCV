import cv2
import sys
import time


if __name__ == '__main__':

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[0]

    if tracker_type == "BOOSTING":
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Video is not opened...")

    ok, frame = cap.read()
    if not ok:
        print("Video is not opened...")
        sys.exit()

    bbox = cv2.selectROI(frame, False)
    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        prev_time = 0

        ok, bbox = tracker.update(frame)

        current_time = time.time()
        fps = current_time/(current_time-prev_time)
        prev_time = current_time

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed..", (100, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.putText(frame, tracker_type, (100, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(fps), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF ==  ord('q'):
            break


