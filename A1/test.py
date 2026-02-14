import cv2

def test_all_indices():
    # We test 0 through 4 because WSL often shifts indices
    for index in range(5):
        print(f"Testing index {index}...")
        # cv2.CAP_V4L2 is the critical flag for WSL/Linux
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ SUCCESS: Camera found at index {index}")
                cap.release()
                return index
            cap.release()
        else:
            print(f"❌ Index {index} is not available.")
    return None

working_id = test_all_indices()

if working_id is not None:
    print(f"Use cv2.VideoCapture({working_id}, cv2.CAP_V4L2) in your project.")
else:
    print("Could not find a working camera stream.")