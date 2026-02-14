import cv2 # OpenCV - Only for video capture and display
import argparse
from utils_new import * # Define custom CV functions in utils.py

def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    
    args = parser.parse_args()
    
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = convert_to_grayscale(frame)
        binary = apply_threshold(gray, 200)
        binary_01 = convert_binary(gray, 200)
        #binary_01 = manual_morph(binary_01, 'dilate')
        #binary_01 = manual_morph(binary_01, 'erode')

        contours,parent,border_type= detect_contours(binary_01)
        #print(parent)
        #contour_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) # Create a black image with the same dimensions as the frame
        candidates = extract_tag_candidates(contours, parent, epsilon_factor=0.02)
        for candidate in candidates:
            # Assuming corners is a numpy array: [[y1, x1], [y2, x2], ...]
            candidate = candidate[:, ::-1]
            src_corners = sort_corners(candidate)
            warped_img = warp_tag(binary, src_corners)
            tag_id, rotations = identify_tag(warped_img)
            print(rotations)
            #print(tag_id)
            if tag_id is not None:
                template_img = cv2.imread(args.template)
                frame = overlay_2d(frame, template_img, src_corners, rotations)
                              
                #cv2.putText(frame, f"ID: {tag_id}", tuple(src_corners[0].astype(int)),    
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    # Display ID on the frame (basic OpenCV utility allowed)
        
        
        #draw_contours(contour_img, contours, color=(0, 255, 0))

        #cv2.imwrite('contours.jpg', contour_img)
        #cv2.imwrite('warped.jpg', warped_img)
        #cv2.imwrite('result.jpg', binary)
        cv2.imshow('Frame', frame)
        #break
        #corners = detect_tag_corners(binary)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
