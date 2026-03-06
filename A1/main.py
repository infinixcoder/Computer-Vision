import cv2 # OpenCV - Only for video capture and display
import argparse
from utils import * # Define custom CV functions in utils.py

def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    
    args = parser.parse_args()

    output_path = 'AR_Detection_Output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30.0  # Adjust based on your webcam or input video fps
    frame_size = (1920, 1080) # Must match your frame size exactly
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return
    
    task = 1
    if(args.template is not None):
        template_img = cv2.imread(args.template)
        task = 2
    elif(args.model is not None):
        obj_model = OBJ(args.model, swapyz=False)
        K = np.loadtxt(r"./camera_matrix.txt")
        task = 3
    else:
        task = 1

        
    frame_no = 1
    while cap.isOpened():
        #print(f"Frame no : {frame_no}")
        frame_no += 1
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = convert_to_grayscale(frame)
        th=kittler_threshold(gray)
        binary_01 = convert_binary(gray, th)
        binary_01 = erosion(binary_01)
        binary = bin_to_img(binary_01)
        cv2.imwrite('result.jpg', binary)


        contours,parent,border_type= detect_contours(binary_01)
        #contour_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) # Create a black image with the same dimensions as the frame
        
        #draw_contours(contour_img, contours, color=(0, 255, 0))
        #cv2.imwrite('contours.jpg', contour_img)

        candidates = extract_tag_candidates(binary, contours, parent, epsilon_factor=0.02)

        

        for candidate in candidates:
            src_corners = sort_corners(candidate)
            warped_img, H = warp_tag(binary, src_corners)
            tag_id, rotations = identify_tag(warped_img)

            if tag_id is not None:
                if(task == 1):
                    orientation_angle = get_continuous_orientation(src_corners, rotations)
                    print(f"Detected Tag ID: {tag_id}, Orientation: {orientation_angle:.2f} degrees")
                    print(f"Tag Corners (clockwise from top-left): {src_corners}")
                    print("--------------------------------------------------")
                    cv2.putText(frame, f"ID: {tag_id}", tuple(src_corners[0].astype(int)),    
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    # Display ID on the frame (basic OpenCV utility allowed)    
                
                elif(task == 2):
                    frame = overlay_2d(frame, template_img, src_corners, rotations)

                else:
                    R, t = decompose_homography(H, K)
                    extrinsic = np.column_stack((R, t))
                    projection_matrix = np.dot(K, extrinsic)
                    frame = render(frame, obj_model, projection_matrix, warped_img)       
                
        
        
        #draw_contours(contour_img, contours, color=(0, 255, 0))

        #cv2.imwrite('contours.jpg', contour_img)
        #cv2.imwrite('warped.jpg', warped_img)
        cv2.namedWindow("AR Tag Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AR Tag Detection", 960, 540)
        out.write(frame)
        cv2.imshow('AR Tag Detection', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting and saving video...")
            break
    
    out.release()       
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
