import cv2
import numpy as np
import math

"""
    Direction Reference:

                1
                |
                |
         0 <- - | - -> 2
                |
                |
                3
    """


def convert_to_grayscale(img):
    return (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]).astype(np.uint8)

def apply_threshold(gray_img, threshold_val):
    return (gray_img > threshold_val).astype(np.uint8) * 255

def convert_binary(gray_img, threshold_val):
    return (gray_img > threshold_val).astype(int)

def manual_morph(img, operation='dilate'):
    # Pad to handle edges
    padded = np.pad(img, 1, mode='constant', constant_values=0)
    result = np.zeros_like(img)
    
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            neighborhood = padded[i-1:i+2, j-1:j+2]
            if operation == 'dilate':
                # If ANY pixel in 3x3 is 1, result is 1
                if np.any(neighborhood == 1):
                    result[i-1, j-1] = 1
            elif operation == 'erode':
                # If ALL pixels in 3x3 are 1, result is 1
                if np.all(neighborhood == 1):
                    result[i-1, j-1] = 1
    return result

def merge_broken_corners(contour_points, dist_threshold=2):
    """
    If two points in the contour are very close, they are likely
    the same corner split by a gap.
    """
    if len(contour_points) < 2: return contour_points
    
    cleaned = []
    i = 0
    while i < len(contour_points):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % len(contour_points)]
        
        # Calculate Euclidean distance
        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        
        if dist < dist_threshold:
            # Average the two points to create a single corner
            midpoint = [(p1[0]+p2[0])//2, (p1[1]+p2[1])//2]
            cleaned.append(midpoint)
            i += 2 # Skip the next point as it's now merged
        else:
            cleaned.append(p1)
            i += 1
    return np.array(cleaned)


#----------------------------------------------------------------------------------------------------------------

def rotate(pixel, dir):
    x, y = pixel
    end = None
    if(dir == 0):
        r = x-1
        c = y
        new_dir = 1
    elif(dir == 1):
        r = x
        c = y+1
        new_dir = 2
    elif(dir == 2):
        r = x+1
        c = y
        new_dir = 3
        end = [x, y+1]
    else:
        r = x
        c = y-1
        new_dir = 0
    
    return r, c, new_dir, end



def find_border(img, start_pixel, prev_pixel, dir, NBD):
    cur_pixel = list(start_pixel)
    next_pixel = list(prev_pixel)
    end_pixel = None
    track_pixel = list(prev_pixel)
    contour = []
    contour.append(list(cur_pixel))

    #Detect isolated point

    while(img[next_pixel[0]][next_pixel[1]] == 0):
        next_pixel[0], next_pixel[1], dir, end = rotate(cur_pixel, dir)

        if(end != None):
            end_pixel = list(end)
        if(track_pixel == next_pixel):          #Isolated Point
            img[cur_pixel[0]][cur_pixel[1]] = -NBD
            return contour

    if(end_pixel != None and img[end_pixel[0]][end_pixel[1]] == 0):   #isolated point
        img[cur_pixel[0]][cur_pixel[1]] = -NBD
        return contour
    elif((end_pixel == None or (end_pixel != None and img[end_pixel[0]][end_pixel[1]] != 0)) and img[cur_pixel[0]][cur_pixel[1]] == 1):
        img[cur_pixel[0]][cur_pixel[1]] = NBD
    else: pass

    prev_pixel = list(cur_pixel)
    cur_pixel = list(next_pixel)
    contour.append(list(cur_pixel))
    first_found_pixel = list(cur_pixel)         # To check when the loop completes
    flag = 0                                    

    if(dir >= 2): dir -= 2  #reverse the direcion as we start rotation from the prev pixel
    else : dir += 2

    while True : 
        if(cur_pixel == first_found_pixel and prev_pixel == start_pixel and flag == 1):
            break
        else:
            flag = 1
            next_pixel[0], next_pixel[1], dir, end = rotate(cur_pixel, dir)

            if(end != None):
                end_pixel = list(end)
            
            while(img[next_pixel[0]][next_pixel[1]] == 0):
                next_pixel[0], next_pixel[1], dir, end = rotate(cur_pixel, dir)
                if(end != None):
                    end_pixel = list(end)
            
            if(end_pixel != None and img[end_pixel[0]][end_pixel[1]] == 0):
                img[cur_pixel[0]][cur_pixel[1]] = -NBD
                end_pixel = None
            elif((end_pixel == None or (end_pixel != None and img[end_pixel[0]][end_pixel[1]] != 0)) and img[cur_pixel[0]][cur_pixel[1]] == 1):
                img[cur_pixel[0]][cur_pixel[1]] = NBD
                end_pixel = None
            else : pass
            prev_pixel = list(cur_pixel)
            cur_pixel = list(next_pixel)
            contour.append(list(cur_pixel))

            if(dir >= 2): dir -= 2  #reverse the direcion as we start rotation from the prev pixel
            else : dir += 2

    return contour



def detect_contours(img):
    padded_img = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    rows, cols = padded_img.shape
    
    LNBD = 1
    NBD = 1
    border_points = []
    parent_border = [-1] # The first index (0) is the background frame
    border_types = [0]   # 0 for holes, 1 for outer

    for i in range(1, rows-1):
        LNBD = 1
        for j in range(1, cols-1):
            border_found = False
            
            # Case 1: Outer Border (White to Black)
            if padded_img[i][j] == 1 and padded_img[i][j-1] == 0:
                NBD += 1
                curr_parent = LNBD
                direction = 0
                border_found = True
                border_types.append(1)
            
            # Case 2: Hole Border (Black to White)
            elif padded_img[i][j] >= 1 and padded_img[i][j+1] == 0:
                NBD += 1
                # If current pixel is already a border, its parent is itself in Suzuki logic
                # But for a standard hierarchy, its parent is the outer border it belongs to
                curr_parent = padded_img[i][j] if padded_img[i][j] > 1 else LNBD
                direction = 0
                border_found = True
                border_types.append(0)

            if border_found:
                parent_border.append(curr_parent)
                contour = find_border(padded_img, [i, j], [i, j-1] if border_types[-1]==1 else [i, j+1], direction, NBD)
                border_points.append(contour)
            
            # Update LNBD for the next pixel in the row
            if abs(padded_img[i][j]) > 1:
                LNBD = abs(padded_img[i][j])

    # 3. Shift contour coordinates back to match the original image space
    # (Subtract 1 from x and y because we added 1 pixel of padding at top/left)
    final_contours = []
    for contour in border_points:
        shifted_contour = [[p[0]-1, p[1]-1] for p in contour]
        final_contours.append(shifted_contour)

    return final_contours, parent_border, border_types


def line_draw(img, start_pixel, end_pixel, color):

    y1, x1 = start_pixel
    y2, x2 = end_pixel

    dx = x2-x1
    dy = y2-y1
    err = 2*dy - dx
    y = y1

    for x in range(x1, x2+1):
        img[y][x] = color
        if(err > 0):
            y += 1
            err = err - 2*dx
        
        err = err + 2*dy
    
    return

def draw_contours(img, contours, color=(0, 255, 0)):
    for contour in contours:
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i+1)%len(contour)]
            line_draw(img, p1, p2, color)
    
    return


#----------------------------------------------------------------------------------------------------------------


def calculate_perimeter(points):
    perimeter = 0
    num_points = len(points)
    for i in range(num_points):
        # Distance between current point and the next (closing the loop at the end)
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        perimeter += dist
    return perimeter


def perpendicular_distance(point, line_start, line_end):

    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    # Distance from point (x0, y0) to line (x1, y1)-(x2, y2)
    # Using the cross-product area formula: distance = |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
    num = np.abs((line_end[1] - line_start[1]) * point[0] - 
                 (line_end[0] - line_start[0]) * point[1] + 
                 line_end[0] * line_start[1] - line_end[1] * line_start[0])
    den = np.sqrt((line_end[1] - line_start[1])**2 + (line_end[0] - line_start[0])**2)
    return num / den


def douglas_peucker(points, epsilon):

    # Start and end points of the current segment
    start_pt = points[0]
    end_pt = points[-1]
    
    # Find the point with the maximum distance from the line segment
    max_dist = 0
    index = 0
    
    # Iterate through points between start and end (exclusive)
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start_pt, end_pt)
        if dist > max_dist:
            max_dist = dist
            index = i
            
    # If the max distance is greater than epsilon, split recursively
    if max_dist > epsilon:
        # Recursive calls for the two halves
        left_half = douglas_peucker(points[:index+1], epsilon)
        right_half = douglas_peucker(points[index:], epsilon)
        
        # Merge results (dropping the duplicate point at the split index)
        return np.vstack((left_half[:-1], right_half))
    else:
        # If no point is far enough, just return the start and end points
        return np.array([start_pt, end_pt])


def get_contour_depth(index, parent_array):
    """Recursively calculates how many parents a contour has."""
    depth = 0
    current = index+1
    # -1 usually indicates the frame or no parent
    while parent_array[current] != -1:
        depth += 1
        current = parent_array[current] - 1
    return depth


def extract_tag_candidates(contours, parent_array, epsilon_factor=0.02):
    candidates = []
    
    for i in range(len(contours)):
        if get_contour_depth(i, parent_array) == 3:
            c = merge_broken_corners(contours[i])
            perimeter = calculate_perimeter(c)
            epsilon = epsilon_factor * perimeter
            approx = douglas_peucker(c, epsilon)
            approx = approx[:-1]
            #print(approx)
            # Step 3: Verify it has exactly 4 corners
            if len(approx) == 4:
                candidates.append(approx)
                
    return candidates


#----------------------------------------------------------------------------------------------------------------

def sort_corners(corners):
    """
    Sorts 4 unique corners in a consistent clockwise order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
   
    pts = corners.reshape(4, 2)

    # Initialize sorted array
    rect = np.zeros((4, 2), dtype="float32")

    # 2. Top-Left has the smallest sum (x + y)
    # 3. Bottom-Right has the largest sum (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]


    # 4. Top-Right has the smallest difference (y - x)
    # 5. Bottom-Left has the largest difference (y - x)
    # np.diff calculates [y - x]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def calculate_homography(src_pts, dst_pts):
    """
    Computes Homography matrix H such that H * src = dst
    src_pts/dst_pts: 4x2 numpy arrays
    """
    A = []
    for i in range(4):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        # Two equations per point correspondence
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    A = np.array(A)
    # Solve Ah = 0 using SVD. h is the last column of V (or last row of V_transpose)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))
    
    # Normalize the matrix
    return H / H[2, 2]


def warp_tag(img, src_corners, output_size=64):
    """
    Warps the tag area into a square output image.
    output_size should be a multiple of 8 (e.g., 160 means 20px per cell).
    """
    # 1. Define destination corners for a square
    dst_corners = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ])
    
    # 2. Compute H mapping frame -> square (or inverse)
    # We calculate H from Square -> Frame to use for pixel pulling
    H = calculate_homography(dst_corners, src_corners)
    
    warped_img = np.zeros((output_size, output_size), dtype=np.uint8)

    # 3. Manual Warp (Pixel Pulling)
    for y in range(output_size):
        for x in range(output_size):
            # Transform destination coordinates back to source
            pos = np.dot(H, [x, y, 1])
            src_x = int(pos[0] / pos[2])
            src_y = int(pos[1] / pos[2])
            
            # Boundary check
            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                warped_img[y, x] = img[src_y, src_x]
                
    return warped_img


def identify_tag(warped_img):
    # Divide into 8x8 grid
    cell_size = warped_img.shape[0] // 8
    grid = np.zeros((8, 8), dtype=int)
    
    for i in range(8):
        for j in range(8):
            # Sample the center of the cell to avoid edge noise
            cell = warped_img[i*cell_size : (i+1)*cell_size, j*cell_size : (j+1)*cell_size]
            center_val = np.mean(cell[cell_size//4 : 3*cell_size//4, cell_size//4 : 3*cell_size//4])
            grid[i, j] = 1 if center_val > 150 else 0 # 1 for White, 0 for Black

    # The internal 4x4 grid is at indices [2:6, 2:6]
    inner_grid = grid[2:6, 2:6]
    
    # Find Orientation Marker (White cell at bottom-right of 4x4)
    # Possible orientations: 0, 90, 180, 270 degrees
    # Corner indices of 4x4: (0,0), (0,3), (3,3), (3,0)
    rotations = 0
    if inner_grid[3, 3] == 1: rotations = 0
    elif inner_grid[3, 0] == 1: rotations = 1 # 90 deg
    elif inner_grid[0, 0] == 1: rotations = 2 # 180 deg
    elif inner_grid[0, 3] == 1: rotations = 3 # 270 deg
    else: return None, None # Marker not found
    
    # Rotate the inner grid to normalize orientation
    oriented_inner = np.rot90(inner_grid, k=rotations)
    
    # Decode ID from central 2x2 (indices 1,2 of the 4x4)
    # Pattern: Clockwise [cite: 43]
    # (1,1) -> (1,2) -> (2,2) -> (2,1)
    b1 = oriented_inner[1, 1]
    b2 = oriented_inner[1, 2]
    b3 = oriented_inner[2, 2]
    b4 = oriented_inner[2, 1]
    
    tag_id = (b1 << 3) | (b2 << 2) | (b3 << 1) | b4
    return tag_id, rotations


#----------------------------------------------------------------------------------------------------------------

def overlay_2d(frame, template, tag_corners, orientation_rotations):
    """
    Superimposes a template image onto the detected AR tag.
    """
    h_temp, w_temp = template.shape[:2]
    
    # 1. Define Template Corners (Clockwise: TL, TR, BR, BL)
    template_pts = np.array([
        [0, 0],
        [w_temp - 1, 0],
        [w_temp - 1, h_temp - 1],
        [0, h_temp - 1],
        
    ], dtype="float32")

    # 2. Align Tag Corners based on orientation from Task 1
    # orientation_rotations is 0, 1, 2, or 3 (k * 90 degrees)
    # We roll the array so the Template's TL matches the Tag's actual TL
    target_pts = np.roll(tag_corners, -orientation_rotations, axis=0)

    # 3. Calculate H mapping Template -> Frame
    # Use your manual DLT function from Task 1
    H = calculate_homography(template_pts, target_pts)
    H_inv = np.linalg.inv(H)

    # 4. Find Bounding Box in the frame to optimize the loop
    min_x = int(np.min(target_pts[:, 0]))
    max_x = int(np.max(target_pts[:, 0]))
    min_y = int(np.min(target_pts[:, 1]))
    max_y = int(np.max(target_pts[:, 1]))

    # Ensure bounds are within frame dimensions
    h_frame, w_frame = frame.shape[:2]
    min_x, max_x = max(0, min_x), min(w_frame - 1, max_x)
    min_y, max_y = max(0, min_y), min(h_frame - 1, max_y)

    # 5. Manual Pixel Pulling Loop
    # We iterate over the frame area and pull from the template
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Transform frame (x,y) to template (u,v)
            pos = np.dot(H_inv, [x, y, 1])
            u = pos[0] / pos[2]
            v = pos[1] / pos[2]

            # If the calculated coordinate falls inside the template, copy the color
            if 0 <= u < w_temp and 0 <= v < h_temp:
                frame[y, x] = template[int(v), int(u)]

    return frame

#----------------------------------------------------------------------------------------------------------------
def generate_tag(cell_size=50, tag_id=0):
    """
    Generate an AR tag image with the specified ID.
    """
    # Initialize an 8x8 black grid (0 = black)
    # The 2-cell outer border is already black by default
    grid = np.zeros((8, 8), dtype=np.uint8)
    
    # Define the internal 4x4 grid (Indices 2 to 5)
    # Row 2
    grid[2, 2] = 0
    grid[2, 3] = 255
    grid[2, 4] = 255
    grid[2, 5] = 0
    
    # Row 3
    grid[3, 2] = 255
    grid[3, 3] = 255  # ID Bit 1
    grid[3, 4] = 0  # ID Bit 2
    grid[3, 5] = 255
    
    # Row 4
    grid[4, 2] = 255
    grid[4, 3] = 255  # ID Bit 4
    grid[4, 4] = 255  # ID Bit 3
    grid[4, 5] = 255
    
    # Row 5
    grid[5, 2] = 255
    grid[5, 3] = 255
    grid[5, 4] = 255
    grid[5, 5] = 0

    # Scale the 8x8 grid to a visible image size
    tag_image = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    
    cv2.imwrite(f"Tag{tag_id}.png", tag_image)

    return tag_image

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame.

    Args:
        img: The current video frame.
        obj: The loaded OBJ model.
        projection: The 3D projection matrix.
        model: The reference image representing the surface to be augmented.
        color: Whether to render in color. Defaults to False.
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img