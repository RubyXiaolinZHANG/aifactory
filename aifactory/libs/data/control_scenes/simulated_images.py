import os

import cv2
import numpy as np

SIMULATED_SENSES = ["solid_color", "gray_gradient", "color_gradient", "stripe", "sin", "edge"]


def generate_solid_color_image(im_height, im_width, color=(128, 128, 128), dtype=np.uint8):
    image = np.ones((im_height, im_width, 3), dtype=dtype) * np.array(color, dtype=dtype).reshape(1, 1, 3)
    return image


def generate_linear_gradient(im_height, im_width, angle_deg,
                             start_color=0, end_color=255, dtype=np.uint8):
    angle_rad = np.deg2rad(angle_deg)
    x = np.arange(im_width)
    y = np.arange(im_height)
    X, Y = np.meshgrid(x, y)
    proj = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    proj_min, proj_max = proj.min(), proj.max()
    proj_norm = (proj - proj_min) / (proj_max - proj_min)
    gradient = start_color + (end_color - start_color) * proj_norm
    return gradient.astype(dtype)


def generate_color_gradient(im_height, im_width, angle_deg,
                            start_color=(0, 0, 0), end_color=(255, 255, 255), dtype=np.uint8):
    grad = generate_linear_gradient(im_height, im_width, angle_deg, 0, 1, dtype=np.float32)
    grad = grad[..., np.newaxis]  # (H,W,1)
    start = np.array(start_color, dtype=dtype).reshape(1, 1, 3)
    end = np.array(end_color, dtype=dtype).reshape(1, 1, 3)
    color_grad = (start * (1 - grad) + end * grad).astype(dtype)
    return color_grad


def generate_knife_edge(im_height, im_width, angle_deg,
                        offset=0.5, color_1=(0, 0, 0), color_2=(255, 255, 255), blur_sigma=0, dtype=np.uint8):
    x = np.arange(im_width)
    y = np.arange(im_height)
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(angle_deg)
    proj = X * np.cos(theta) + Y * np.sin(theta)
    proj_min, proj_max = proj.min(), proj.max()
    threshold = proj_min + offset * (proj_max - proj_min)
    mask = (proj >= threshold)[..., np.newaxis].astype(dtype)  # .astype(np.uint8) * 255
    edge = (np.array(color_1, dtype=dtype).reshape(1, 1, 3) * mask +
            np.array(color_2, dtype=dtype).reshape(1, 1, 3) * (1 - mask))
    if blur_sigma > 0:
        edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=blur_sigma)

    return edge


def generate_stripes_0(im_height, im_width, angle_deg, period, stripe_type='sin', phase=0, color_1=(0, 0, 0),
                     color_2=(255, 255, 255), dtype=np.uint8):
    x = np.arange(im_width)
    y = np.arange(im_height)
    X, Y = np.meshgrid(x, y)
    normal_angle = angle_deg + 90
    theta = np.deg2rad(normal_angle)
    proj = X * np.cos(theta) + Y * np.sin(theta)
    if stripe_type == 'sin':
        intensity = (0.5 + 0.5 * np.sin(2 * np.pi * proj / period + phase))[..., np.newaxis]

    elif stripe_type == 'square':
        intensity = (np.sin(2 * np.pi * proj / period + phase) >= 0)[..., np.newaxis]
    else:
        raise ValueError("stripe_type should be 'sin' or 'square'")
    stripe = (intensity * np.array(color_1, dtype=dtype).reshape(1, 1, 3) +
              (1 - intensity) * np.array(color_2, dtype=dtype).reshape(1, 1, 3)).astype(dtype)
    return stripe

def generate_stripes(im_height, im_width, angle_deg, period,  phase=0, stripe_type='square', bg=(0, 0, 0),
                     fg=(255, 255, 255), dtype=np.uint8):
    """
    Generate a stripe target with specified period and orientation.

    Parameters:
        im_height (int): Image height (pixels)
        im_width (int): Image width (pixels)
        period (float): Stripe period (pixels per cycle), i.e., distance between centers of two adjacent same-color stripes
        angle (float): Orientation angle of the stripes (degrees). 0° means horizontal stripes (lines are horizontal, intensity varies vertically),
                       90° means vertical stripes (lines are vertical, intensity varies horizontally)
        phase (float): Phase shift (pixels) to translate the stripe pattern
        stripe_type (str): Stripe type, either 'square' (square wave) or 'sine' (sine wave)
        bg (int): Background/bright grayscale value (0~255)
        fg (int): Foreground/dark grayscale value (0~255)

    Returns:
        numpy.ndarray: Grayscale image of shape (height, width), dtype uint8
    """
    # Create mesh grid coordinates
    x = np.arange(im_width)
    y = np.arange(im_height)
    X, Y = np.meshgrid(x, y)

    # Calculate the angle perpendicular to the stripe direction
    # Stripe line direction is angle degrees; the normal direction is angle + 90 degrees
    theta_perp = np.radians(angle_deg + 90)

    # Compute the projection distance of each pixel onto the normal direction
    # Projection distance d = x * cos(theta) + y * sin(theta)
    d = X * np.cos(theta_perp) + Y * np.sin(theta_perp)

    # Apply phase shift
    d_phase = d + phase

    if stripe_type == 'square':
        # Square wave: determine black/white based on modulo of the distance
        # Take modulo period and check if it's less than half period
        mask = d_phase % period < period / 2
        value = np.zeros((im_height, im_width, 3), dtype=dtype) + np.array(fg)
        value[mask] = np.array(bg)
    elif stripe_type == 'sin':
        # Sine wave: intensity varies sinusoidally
        # Normalize phase to [0, 2π]
        phase_norm = 2 * np.pi * d_phase / period
        # Map sine values from [-1,1] to grayscale range between bg and fg
        # Formula: value = (bg + fg)/2 + ((bg - fg)/2) * sin(phase_norm)
        mean = (np.array(bg) + np.array(fg)).reshape(1,1, 3) / 2
        amp = (np.array(bg) - np.array(fg)).reshape(1,1, 3)  / 2
        value = mean + amp * np.sin(phase_norm)[...,np.newaxis]
        # Clip to [0,255] and convert to integer
        min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
        value = np.clip(value, min_val, max_val).astype(dtype)
    else:
        raise ValueError("stripe_type must be either 'square' or 'sin'")

    return value.astype(dtype)

def generate_rotated_grid(output_height, output_width,  cell_size=50, angle_deg=0,
                          mode='checker', color1=(0,0,0), color2=(255,255,255),
                          palette=None, bg_color=(128,128,128)):
    """
    Generate a rotated grid pattern by direct geometric drawing (no affine transform).
    The output image has the specified dimensions. Cell size is fixed; the number
    of rows and columns is automatically derived to fit into the output canvas.
    The rotated grid is centered in the output image; parts outside the canvas
    are clipped (no interpolation blur).

    Parameters:
        output_width, output_height : Dimensions of the output image (pixels)
        cell_size    : Side length of each cell in pixels
        angle_deg    : Overall rotation angle in degrees (counter‑clockwise)
        mode         : 'checker' for two‑color checkerboard, 'color' for colored grid
        color1,color2: Two colors for checkerboard mode (B,G,R)
        palette      : For 'color' mode – specifies how to assign cell colors:
                       - None          : use default palette (red, green, blue, cyan, magenta, yellow)
                       - list of tuples: custom palette (B,G,R) – repeated cyclically
                       - int           : number of random colors to generate
                       - str           : name of an OpenCV colormap (e.g., 'jet', 'hot', 'rainbow')
        bg_color     : Background color (B,G,R) for areas outside the grid

    Returns:
        rotated_img  : Rotated grid image of size (output_height, output_width, 3), uint8 BGR
    """
    # Determine the number of rows and columns that fit into the output canvas
    cols = output_width // cell_size
    rows = output_height // cell_size
    if cols == 0 or rows == 0:
        raise ValueError(f"cell_size {cell_size} too large for output dimensions "
                         f"{output_width}x{output_height}; at least one cell must fit.")

    # Dimensions of the original (unrotated) grid
    orig_w = cols * cell_size
    orig_h = rows * cell_size
    orig_cx = orig_w / 2.0
    orig_cy = orig_h / 2.0

    # Centers of output canvas
    out_cx = output_width / 2.0
    out_cy = output_height / 2.0

    # Precompute rotation parameters
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    # Build output coordinate grid (origin at top-left)
    y_out, x_out = np.mgrid[:output_height, :output_width].astype(np.float32)

    # Shift to output center as origin
    x_centered = x_out - out_cx
    y_centered = y_out - out_cy

    # Apply inverse rotation to get coordinates in the original (unrotated) grid space
    x_orig = c * x_centered + s * y_centered + orig_cx
    y_orig = -s * x_centered + c * y_centered + orig_cy

    # Compute cell indices (row = i, column = j)
    j = np.floor(x_orig / cell_size).astype(np.int32)
    i = np.floor(y_orig / cell_size).astype(np.int32)

    # Mask for pixels inside the original grid bounds
    valid = (j >= 0) & (j < cols) & (i >= 0) & (i < rows)

    # Initialize output image with background color
    img = np.full((output_height, output_width, 3), bg_color, dtype=np.uint8)

    if mode == 'checker':
        # Checkerboard: color based on parity of (i + j)
        parity = (i + j) % 2
        mask1 = valid & (parity == 0)
        mask2 = valid & (parity == 1)
        img[mask1] = color1
        img[mask2] = color2

    elif mode == 'color':
        # Build color lookup table for all cells (row‑major order)
        total_cells = rows * cols

        if palette is None:
            # Default palette (BGR)
            base_colors = [(255,0,0), (0,255,0), (0,0,255),
                           (255,255,0), (255,0,255), (0,255,255)]
            color_table = np.array([base_colors[i % len(base_colors)] for i in range(total_cells)],
                                   dtype=np.uint8)
        elif isinstance(palette, list):
            # User-supplied list of BGR tuples
            color_table = np.array([palette[i % len(palette)] for i in range(total_cells)],
                                   dtype=np.uint8)
        elif isinstance(palette, int):
            # Generate 'palette' distinct random colors
            rng = np.random.RandomState()
            random_colors = rng.randint(0, 256, (palette, 3), dtype=np.uint8)
            color_table = np.array([random_colors[i % palette] for i in range(total_cells)],
                                   dtype=np.uint8)
        elif isinstance(palette, str):
            # OpenCV colormap
            gray_vals = np.linspace(0, 255, total_cells, dtype=np.uint8).reshape(1, -1)
            colormap_dict = {
                'AUTUMN': cv2.COLORMAP_AUTUMN, 'BONE': cv2.COLORMAP_BONE,
                'JET': cv2.COLORMAP_JET, 'WINTER': cv2.COLORMAP_WINTER,
                'RAINBOW': cv2.COLORMAP_RAINBOW, 'OCEAN': cv2.COLORMAP_OCEAN,
                'SUMMER': cv2.COLORMAP_SUMMER, 'SPRING': cv2.COLORMAP_SPRING,
                'COOL': cv2.COLORMAP_COOL, 'HSV': cv2.COLORMAP_HSV,
                'PINK': cv2.COLORMAP_PINK, 'HOT': cv2.COLORMAP_HOT,
                'PARULA': cv2.COLORMAP_PARULA, 'MAGMA': cv2.COLORMAP_MAGMA,
                'INFERNO': cv2.COLORMAP_INFERNO, 'PLASMA': cv2.COLORMAP_PLASMA,
                'VIRIDIS': cv2.COLORMAP_VIRIDIS, 'CIVIDIS': cv2.COLORMAP_CIVIDIS,
                'TWILIGHT': cv2.COLORMAP_TWILIGHT, 'TWILIGHT_SHIFTED': cv2.COLORMAP_TWILIGHT_SHIFTED,
                'TURBO': cv2.COLORMAP_TURBO, 'DEEPGREEN': cv2.COLORMAP_DEEPGREEN,
            }
            cmap = colormap_dict.get(palette.upper())
            if cmap is None:
                raise ValueError(f"Unsupported colormap: {palette}. Available: {list(colormap_dict.keys())}")
            colored = cv2.applyColorMap(gray_vals, cmap)  # shape (1, total_cells, 3)
            color_table = colored[0]  # (total_cells, 3)
        else:
            raise TypeError("palette must be None, list, int, or str")

        # Compute linear index for each valid pixel
        linear_idx = i[valid] * cols + j[valid]
        img[valid] = color_table[linear_idx]

    else:
        raise ValueError("mode must be 'checker' or 'color'")

    return img


def generate_resolution_circle(im_height, im_width, margin=20, start_width=1, width_increment=1,
                             color_1=(255, 255, 255), color_2=(0, 0, 0)):
    """
       Generate concentric rings with variable spacing for resolution testing.

       Parameters:
           width, height: Image dimensions (pixels)
           margin: Minimum distance from target center to image border (pixels)
           start_width: Width of the outermost ring (pixels), usually set to 1 to test limiting resolution
           width_increment: Increase in ring width for each inner ring (pixels), controls the rate of width change
           color_1: First color (BGR), used for the outermost ring
           color_2: Second color (BGR), alternates with color1

       Returns:
           numpy.ndarray: BGR image
       """
    # Create background image, initialized with color1
    img = np.full((im_height, im_width, 3), color_1, dtype=np.uint8)
    center = (im_width // 2, im_height // 2)

    # Calculate maximum allowed radius (consider margin)
    max_radius = min(center[0], center[1], im_width - center[0], im_height - center[1]) - margin
    if max_radius <= 0:
        raise ValueError("Image too small or margin too large, cannot draw valid pattern")

    # Generate list of radius boundaries from outer to inner, ring width increases gradually
    radii = [max_radius]  # outermost boundary
    current_r = max_radius
    w = start_width
    while True:
        next_r = current_r - w
        if next_r <= 0:
            radii.append(0)  # center as the last boundary
            break
        radii.append(next_r)
        current_r = next_r
        w += width_increment

    # Assign colors to each radius boundary (alternating from outer to inner)
    # Note: colors list length equals radii length, each color corresponds to drawing a solid circle with that radius
    colors = []
    for i in range(len(radii)):
        colors.append(color_1 if i % 2 == 0 else color_2)

    # Draw solid circles from largest to smallest radius; later circles cover inner areas, forming rings
    for r, col in zip(radii, colors):
        cv2.circle(img, center, r, col, thickness=-1)
    return img


def create_siemens_star(im_height, im_width, num_wedges=36, bg_color=255, fg_color=0, circle_only=True):
    """
    Generate a circular radial resolution target (Siemens star).

    Parameters:
        size (int): Image width and height (square)
        num_wedges (int): Total number of black and white stripes (must be even)
        bg_color (int): Background grayscale value (0~255); if color image, pass a tuple
        fg_color (int): Foreground grayscale value (0~255); alternates with bg_color
        circle_only (bool): Whether to keep stripes only within the circular area, filling outside with background color

    Returns:
        numpy.ndarray: Grayscale image (or color image if bg_color/fg_color are tuples)
    """
    # Determine if color mode
    is_color = isinstance(bg_color, (tuple, list)) or isinstance(fg_color, (tuple, list))
    if is_color:
        # Ensure both colors are tuples
        if not isinstance(bg_color, (tuple, list)):
            bg_color = (bg_color, bg_color, bg_color)
        if not isinstance(fg_color, (tuple, list)):
            fg_color = (fg_color, fg_color, fg_color)
        img = np.zeros((im_height, im_width, 3), dtype=np.uint8)
    else:
        img = np.zeros((im_height, im_width), dtype=np.uint8)

    # Center coordinates
    cx, cy = im_width // 2, im_height // 2
    # Maximum radius (inscribed circle radius of the image)
    max_r = min(cx, cy)

    # Generate grid coordinates
    y, x = np.ogrid[:im_height, :im_width]
    # Compute polar coordinates relative to center
    dx = x - cx
    dy = y - cy
    r = np.hypot(dx, dy)          # radius
    theta = np.arctan2(dy, dx)     # angle in [-π, π]
    theta = np.where(theta < 0, theta + 2*np.pi, theta)  # convert to [0, 2π)

    # Angular width of each wedge
    wedge_width = 2 * np.pi / num_wedges

    # Determine which wedge index each pixel belongs to (0 ~ num_wedges-1)
    wedge_index = (theta // wedge_width).astype(int)
    # Color pattern based on wedge index parity (even -> bg_color, odd -> fg_color)
    color_pattern = (wedge_index % 2 == 0)   # True -> bg_color, False -> fg_color

    # Fill image according to color mode
    if is_color:
        # Color fill
        for c in range(3):
            img[:,:,c] = np.where(color_pattern, bg_color[c], fg_color[c])
    else:
        # Grayscale fill
        img = np.where(color_pattern, bg_color, fg_color).astype(np.uint8)

    # If only the circular area is kept, set pixels outside the circle to background color
    if circle_only:
        mask = r <= max_r
        if is_color:
            # Apply mask to each channel
            for c in range(3):
                img[:,:,c] = np.where(mask, img[:,:,c], bg_color[c])
        else:
            img = np.where(mask, img, bg_color).astype(np.uint8)

    return img
####################################################################################################################
# the followings are test code


def test_solid_color():
    image = generate_solid_color_image(128, 256, dtype=np.uint8)
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/1_solid_color.png", image)


def test_linear_gradient():
    image = generate_linear_gradient(128, 256, 45, 0, 255)
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/2_linear_gradient.png", image)


def test_color_gradient():
    start_color = [255, 0, 0]
    end_color = [0, 0, 255]
    image = generate_color_gradient(128, 256, 45, start_color=start_color, end_color=end_color)
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/3_color_gradient.png", image)


def test_knife_edge():
    image = generate_knife_edge(128, 256, 120, color_1=(0, 255, 255), color_2=(255, 0, 0))
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/4_knife_edge.png", image)


def test_stripes():
    image = generate_stripes(128, 256, 0, 4,  phase=0, stripe_type='sin', bg=(0, 0, 0),
                     fg=(255, 255, 255))
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/5_strip_vertical_sin.png", image)
    image = generate_stripes(128, 256, 45, 2, phase=0, stripe_type='square', bg=(0, 0, 0),
                     fg=(255, 0, 255))
    cv2.imwrite("test/5_square_45.png", image)


def test_grid():
    image = generate_rotated_grid(128, 256, cell_size=64, angle_deg=30,
                                 mode='checker', color1=(0, 0, 0), color2=(255, 255, 255))
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/6_checker.png", image)

    image = generate_rotated_grid(128, 256, cell_size=32, angle_deg=45,
                                 mode='color', palette='jet')
    cv2.imwrite("test/6_jet.png", image)

    image = generate_rotated_grid(128, 256, cell_size=80, angle_deg=-15,
                                 mode='color', palette=10)
    cv2.imwrite("test/6_random_color.png", image)


def test_resolution_circle():
    image = generate_resolution_circle(128, 256, margin=10,
                                      start_width=0, width_increment=1,
                                      color_1=(255, 255, 255), color_2=(0, 0, 0))
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/7_resolution_circle.png", image)


def test_siemens_star():
    image = create_siemens_star(128, 256, num_wedges=36, bg_color=[0, 255, 255], fg_color=[0, 0, 255], circle_only=True)
    os.makedirs("test", exist_ok=True)
    cv2.imwrite("test/8_siemens_star.png", image)

if __name__ == "__main__":
    # 1. test solid color
    test_solid_color()
    # 2. test linear gradient
    test_linear_gradient()
    # 3. test color gradient
    test_color_gradient()
    # 4. test knife edge
    test_knife_edge()
    # 5. test stripes
    test_stripes()
    # 6. test grid
    test_grid()
    # 7. test circle resolution target
    test_resolution_circle()
    # 8. test
    test_siemens_star()