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
    rows, cols = img.shape
    print(rows, cols)
    LNBD = 1
    NBD = 1
    border_points = []
    parent_border = []
    parent_border.append(-1)
    border_type = []
    border_type.append(0)

    for i in range(1,rows-1):
        LNBD=1
        for j in range(1,cols-1):
            if img[i][j]==1 and img[i][j-1]==0:
                NBD+=1
                direction=0
                parent_border.append(LNBD) 
                contour= find_border(img,[i,j],[i,j-1],direction,NBD)
                border_points.append(contour)
                border_type.append(1)
                if border_type[NBD-2]==1: parent_border.append(parent_border[NBD-2])
                else:
                    if img[i][j]!=1: LNBD=abs(img[i][j])              
            elif img[i][j]>=1 and img[i][j+1]==0:
                NBD+=1
                direction=0
                if img[i][j]>1: LNBD=img[i][j]
                parent_border.append(LNBD)
                contour = find_border(img,[i,j],[i,j+1],direction,NBD)
                border_points.append(contour)
                border_type.append(0)
                if border_type[NBD-2]==0: parent_border.append(parent_border[NBD-2])
                else:
                    if img[i][j]!=1: LNBD=abs(img[i][j])




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