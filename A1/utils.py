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

def bin_to_img(img):
    return (img).astype(np.uint8)*255

def convert_to_grayscale(img):
    return (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]).astype(np.uint8)

def apply_threshold(gray_img, threshold_val):
    return (gray_img > threshold_val).astype(np.uint8) * 255


def erosion(binary_img):
    shifts = [
        binary_img,
        np.roll(binary_img,  1, axis=0),
        np.roll(binary_img, -1, axis=0),
        np.roll(binary_img,  1, axis=1),
        np.roll(binary_img, -1, axis=1),
        np.roll(np.roll(binary_img,  1, axis=0),  1, axis=1),
        np.roll(np.roll(binary_img,  1, axis=0), -1, axis=1),
        np.roll(np.roll(binary_img, -1, axis=0),  1, axis=1),
        np.roll(np.roll(binary_img, -1, axis=0), -1, axis=1),
    ]
    return np.minimum.reduce(shifts)

def kittler_threshold(gray_img):
    hist, _ = np.histogram(gray_img, bins=256, range=(0, 255))
    probs = hist.astype(float)
    total = probs.sum()
    if total == 0:
        return 0
    probs /= total
    eps = 1e-12

    indices = np.arange(256, dtype=float)
    c_p = np.cumsum(probs)                    # cumulative probability
    c_px = np.cumsum(probs * indices)         # cumulative first moment
    c_px2 = np.cumsum(probs * (indices**2))   # cumulative second moment
    total_mean = c_px[-1]
    total_px2 = c_px2[-1]

    # Evaluate for thresholds t = 1 .. 254 (matching original loop excluding 0 and 255)
    t_idx = np.arange(1, 255)
    p1 = c_p[t_idx]
    p2 = 1.0 - p1

    valid = (p1 > eps) & (p2 > eps)
    if not np.any(valid):
        return 0

    mu1 = np.zeros_like(p1); mu2 = np.zeros_like(p1)
    E1_sq = np.zeros_like(p1); E2_sq = np.zeros_like(p1)

    mu1[valid] = c_px[t_idx][valid] / p1[valid]
    mu2[valid] = (total_mean - c_px[t_idx][valid]) / p2[valid]

    E1_sq[valid] = c_px2[t_idx][valid] / p1[valid]
    E2_sq[valid] = (total_px2 - c_px2[t_idx][valid]) / p2[valid]

    var1 = np.maximum(1e-6, E1_sq - mu1**2)
    var2 = np.maximum(1e-6, E2_sq - mu2**2)

    # Kittler criterion vectorized; set large value where invalid
    J = np.full_like(p1, np.inf, dtype=float)
    v = valid
    J[v] = 1.0 + 2.0 * (p1[v] * np.log(np.sqrt(var1[v])) + p2[v] * np.log(np.sqrt(var2[v]))) \
           - 2.0 * (p1[v] * np.log(p1[v]) + p2[v] * np.log(p2[v]))

    best_rel = np.argmin(J)
    best_T = int(t_idx[best_rel])
    return best_T

def convert_binary(gray_img, threshold_val):
    return (gray_img > threshold_val).astype(int)

def merge_broken_corners(contour_points, dist_threshold=2):
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

def sort_corners(corners):
    """
    Sorts 4 unique corners in a consistent clockwise order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
   
    pts = corners.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def tag_not_paper(image, corners, sample_dist=3):
    
    center = np.mean(corners, axis=0)
    
    # 2. Sample points slightly 'inward' from each corner toward the center
    sample_points = []
    for corner in corners:
        vec = center - corner
        unit_vec = vec / np.linalg.norm(vec)
        p = (corner + unit_vec * sample_dist).astype(int)
        sample_points.append(image[p[1], p[0]]) 
        
    avg_intensity = np.mean(sample_points)
    return avg_intensity < 100

def is_contour_convex(contour):

    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return False

    signs = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        
        u = p2 - p1
        v = p3 - p2

        # 2D Cross Product: (ux * vy) - (uy * vx)
        cp = u[0] * v[1] - u[1] * v[0]
        
        if cp != 0:
            signs.append(np.sign(cp))
            
    # If all non-zero cross products have the same sign, it is convex
    if not signs:
        return True # Collinear points
        
    return all(s == signs[0] for s in signs)

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

    start_pt = points[0]
    end_pt = points[-1]
    
    max_dist = 0
    index = 0
    
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start_pt, end_pt)
        if dist > max_dist:
            max_dist = dist
            index = i
            
    # If the max distance is greater than epsilon, split recursively
    if max_dist > epsilon:
        left_half = douglas_peucker(points[:index+1], epsilon)
        right_half = douglas_peucker(points[index:], epsilon)
        
        # Merge results (dropping the duplicate point at the split index)
        return np.vstack((left_half[:-1], right_half))
    else:
        # If no point is far enough, just return the start and end points
        return np.array([start_pt, end_pt])



def extract_tag_candidates(img, contours, parent_array, epsilon_factor=0.02):
    candidates = []
    
    for i in range(len(contours)):
        
        c = merge_broken_corners(contours[i])
        perimeter = calculate_perimeter(c)
        epsilon = epsilon_factor * perimeter
        approx = douglas_peucker(c, epsilon)
        approx = approx[:-1]
        
        if len(approx) == 4:
            approx = approx[:, ::-1]
            approx = sort_corners(approx)
            
            if(tag_not_paper(img, approx) and is_contour_convex(approx)):   
                candidates.append(approx)
                
    return candidates


#----------------------------------------------------------------------------------------------------------------

def get_continuous_orientation(src_corners, discrete_rot_index):

    oriented_corners = np.roll(src_corners, -discrete_rot_index, axis=0)
    
    p1 = oriented_corners[0] # Top-Left
    p2 = oriented_corners[1] # Top-Right
    
    vector = p2 - p1
    
    angle_rad = np.arctan2(vector[1], vector[0])
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_homography(src_pts, dst_pts):

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


def warp_tag(img, src_corners, output_size=160):

    dst_corners = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=float)

    H = calculate_homography(dst_corners, src_corners)

    xs = np.arange(output_size)
    ys = np.arange(output_size)
    xv, yv = np.meshgrid(xs, ys)  # shape (output_size, output_size)
    ones = np.ones_like(xv, dtype=float)

    coords = np.stack([xv.ravel(), yv.ravel(), ones.ravel()], axis=0)  # (3, N)
    pos = H.dot(coords)  # (3, N)
    pos /= pos[2:3, :]   # normalize

    src_x = np.round(pos[0]).astype(int)
    src_y = np.round(pos[1]).astype(int)

    h_src, w_src = img.shape[:2]
    mask = (src_x >= 0) & (src_x < w_src) & (src_y >= 0) & (src_y < h_src)

    warped = np.zeros(output_size * output_size, dtype=np.uint8)
    if img.ndim == 2:
        src_flat = img[src_y[mask], src_x[mask]]
        warped[mask] = src_flat
    else:
        # preserve grayscale expectation; original used 2D binary images for warp_tag
        warped = warped.reshape((output_size, output_size))

    warped = warped.reshape((output_size, output_size))
    return warped, H


def identify_tag(warped_img):
    # Divide into 8x8 grid
    cell_size = warped_img.shape[0] // 8
    grid = np.zeros((8, 8), dtype=int)
    
    for i in range(8):
        for j in range(8):
            cell = warped_img[i*cell_size : (i+1)*cell_size, j*cell_size : (j+1)*cell_size]
            center_val = np.mean(cell[cell_size//4 : 3*cell_size//4, cell_size//4 : 3*cell_size//4])
            grid[i, j] = 1 if center_val > 150 else 0 

    
    inner_grid = grid[2:6, 2:6]
    
    # Possible orientations: 0, 90, 180, 270 degrees
    # Corner indices of 4x4: (0,0), (0,3), (3,3), (3,0)
    rotations = 0
    if inner_grid[3, 3] == 1: rotations = 0
    elif inner_grid[3, 0] == 1: rotations = 1 
    elif inner_grid[0, 0] == 1: rotations = 2 
    elif inner_grid[0, 3] == 1: rotations = 3 
    else: return None, None # Marker not found
    
    oriented_inner = np.rot90(inner_grid, k=rotations)
    
    b1 = oriented_inner[1, 1]
    b2 = oriented_inner[1, 2]
    b3 = oriented_inner[2, 2]
    b4 = oriented_inner[2, 1]
    
    tag_id = b1 | (b2 << 1) | (b3 << 2) | (b4 << 3)
    return tag_id, rotations


#----------------------------------------------------------------------------------------------------------------

def overlay_2d(frame, template, tag_corners, orientation_rotations):
    h_temp, w_temp = template.shape[:2]

    template_pts = np.array([
        [0, 0],
        [w_temp - 1, 0],
        [w_temp - 1, h_temp - 1],
        [0, h_temp - 1],
    ], dtype="float32")

    target_pts = np.roll(tag_corners, -orientation_rotations, axis=0)
    H = calculate_homography(template_pts, target_pts)
    H_inv = np.linalg.inv(H)

    min_x = int(np.floor(np.min(target_pts[:, 0])))
    max_x = int(np.ceil(np.max(target_pts[:, 0])))
    min_y = int(np.floor(np.min(target_pts[:, 1])))
    max_y = int(np.ceil(np.max(target_pts[:, 1])))

    h_frame, w_frame = frame.shape[:2]
    min_x, max_x = max(0, min_x), min(w_frame - 1, max_x)
    min_y, max_y = max(0, min_y), min(h_frame - 1, max_y)

    if min_x > max_x or min_y > max_y:
        return frame

    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(xs, ys)
    ones = np.ones_like(xv, dtype=float)

    coords = np.stack([xv.ravel(), yv.ravel(), ones.ravel()], axis=0)
    pos = H_inv.dot(coords)
    pos /= pos[2:3, :]

    u = np.round(pos[0]).astype(int)
    v = np.round(pos[1]).astype(int)

    valid = (u >= 0) & (u < w_temp) & (v >= 0) & (v < h_temp)

    if not np.any(valid):
        return frame

    dst_x = coords[0].astype(int)[valid]
    dst_y = coords[1].astype(int)[valid]
    src_u = u[valid]
    src_v = v[valid]

    frame[dst_y, dst_x] = template[src_v, src_u]

    return frame

#----------------------------------------------------------------------------------------------------------------

def decompose_homography(H, K):

    K_inv = np.linalg.inv(K)
    H_inv = np.linalg.inv(H)
    h_prime = np.dot(K_inv, H)
    
    h1 = h_prime[:, 0]
    h2 = h_prime[:, 1]
    h3 = h_prime[:, 2]
    
    # 3. Calculate scaling factor lambda
    lam = 1 / ((np.linalg.norm(h1) + np.linalg.norm(h2)) / 2)
    
    # 4. Extract rotation columns and translation
    r1 = lam * h1
    r2 = lam * h2
    t = lam * h3
    
    # 5. Compute r3 via cross product
    r3 = np.cross(r1, r2)
    
    # 6. Reconstruct Rotation Matrix and ensure it's orthonormal via SVD
    R_raw = np.stack((r1, r2, r3), axis=1)
    U, S, Vt = np.linalg.svd(R_raw)
    R = np.dot(U, Vt)
    
    return R, t

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

def get_face_depth(face, vertices, projection):
    # Get the Z-value of the face after projection to the camera space
    face_vertices = [vertices[i - 1] for i in face[0]]
    avg_z = np.mean([np.dot(projection, [v[0], v[1], v[2], 1])[2] for v in face_vertices])
    return avg_z

def get_object_extents(obj):
    verts = np.array(obj.vertices)
    # Find the min and max coordinates across all vertices
    min_coords = np.min(verts, axis=0)
    max_coords = np.max(verts, axis=0)
    # Calculate the size in each dimension (X, Y, Z)
    obj_width = max_coords[0] - min_coords[0]
    obj_length = max_coords[1] - min_coords[1]
    
    return max(obj_width, obj_length)

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
    DEFAULT_COLOR = (0, 255, 0)
    vertices = obj.vertices
    obj_size = get_object_extents(obj)
    h, w = model.shape[:2]
    # We want the object to take up 80% of the tag width 'w'
    target_pixel_size = w * 0.9
    # The multiplier needed to scale the object to that size
    dynamic_scale = target_pixel_size / obj_size
    
    scale_matrix = np.eye(3) * dynamic_scale
    h, w = model.shape
    light_dir = np.array([1, 1, 1])

    sorted_faces = sorted(obj.faces, key=lambda f: get_face_depth(f, vertices, projection), reverse=True)

    for face in sorted_faces:

        face_vertices = face[0]
        normal_indices = face[1]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        rotated_points = np.array([[p[0], -p[2], -p[1]] for p in points])
        points = np.dot(rotated_points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        face_normal = np.cross(v1, v2)
        norm_val = np.linalg.norm(face_normal)

        if color is False:
            if norm_val > 0:
                face_normal /= norm_val
            # Dot product with light coming from the camera [0, 0, 1]
                brightness = max(0.1, np.dot(face_normal, light_dir))
            else:
                brightness = 0.1
            
            shaded_color = tuple([int(c * brightness) for c in DEFAULT_COLOR])
            cv2.fillConvexPoly(img, imgpts, shaded_color)
        else:
            if norm_val > 0:
                face_normal /= norm_val
            # Dot product with light coming from the camera [0, 0, 1]
                brightness = max(0.1, np.dot(face_normal, light_dir))
            else:
                brightness = 0.1
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            shaded_color = tuple([int(c * brightness) for c in color])
            cv2.fillConvexPoly(img, imgpts, shaded_color)

    return img