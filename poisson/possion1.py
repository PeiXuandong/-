import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F

# Initialize the polygon state
def initialize_polygon():
    return {'points': [], 'closed': False}

# Add a point to the polygon when the user clicks on the image
def add_point(img_original, polygon_state, evt: gr.SelectData):
    if polygon_state['closed']:
        return img_original, polygon_state
    
    x, y = evt.index
    polygon_state['points'].append((x, y))
    
    # Ensure img_original is a PIL.Image.Image
    if isinstance(img_original, np.ndarray):
        img_original = Image.fromarray(img_original)
    
    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)
    
    if len(polygon_state['points']) > 1:
        draw.line(polygon_state['points'], fill='red', width=2)
    
    for point in polygon_state['points']:
        draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill='blue')
    
    return img_with_poly, polygon_state

# Close the polygon when the user clicks the "Close Polygon" button
def close_polygon(img_original, polygon_state):
    if not polygon_state['closed'] and len(polygon_state['points']) > 2:
        polygon_state['closed'] = True
        
        # Ensure img_original is a PIL.Image.Image
        if isinstance(img_original, np.ndarray):
            img_original = Image.fromarray(img_original)
        
        img_with_poly = img_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        draw.polygon(polygon_state['points'], outline='red')
        return img_with_poly, polygon_state
    else:
        return img_original, polygon_state

# Update the background image by drawing the shifted polygon on it
def update_background(background_image_original, polygon_state, dx, dy):
    if background_image_original is not None and polygon_state['closed']:
        # Ensure background_image_original is a PIL.Image.Image
        if isinstance(background_image_original, np.ndarray):
            background_image_original = Image.fromarray(background_image_original)
        
        img_with_poly = background_image_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        shifted_points = [(x + dx, y + dy) for x, y in polygon_state['points']]
        draw.polygon(shifted_points, outline='red')
        return img_with_poly
    else:
        return background_image_original

# Placeholder function for creating a mask from polygon points
def create_mask_from_points(points, img_h, img_w):
    # 将 points 转换为 NumPy 数组
    points = np.array(points)
    
    # 如果第一个点和最后一个点不同，则将第一个点添加到列表的末尾
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # 检查点的顺序和范围
    for point in points:
        if not (0 <= point[0] < img_w and 0 <= point[1] < img_h):
            raise ValueError(f"Point {point} is out of image bounds ({img_w}, {img_h})")
    
    # 创建一个空白的灰度图像
    mask = Image.new('L', (img_w, img_h), 0)
    
    # 创建一个绘图对象
    draw = ImageDraw.Draw(mask)
    
    # 绘制多边形，轮廓和填充都为白色（255）
    draw.polygon(points.flatten().tolist(), outline=255, fill=255)
    
    # 将图像转换为 NumPy 数组
    mask_np = np.array(mask)

    
    return mask_np

# 计算梯度
def compute_gradient(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    
    # 初始化梯度张量
    grad_x = torch.zeros_like(image)
    grad_y = torch.zeros_like(image)
    
    # 对每个通道分别计算梯度
    for channel in range(image.size(1)):
        grad_x[:, channel, :, :] = F.conv2d(image[:, channel:channel+1, :, :], sobel_x, padding=1)
        grad_y[:, channel, :, :] = F.conv2d(image[:, channel:channel+1, :, :], sobel_y, padding=1)
    
    # 将梯度拼接在一起
    return torch.cat([grad_x, grad_y], dim=1)

# 计算像素对
def compute_pixel_pairs(mask):
    # 使用3x3卷积核检测边缘点
    kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(mask.device)
    edge_mask = F.conv2d(mask, kernel, padding=1)
    
    # 边缘点的条件：自身为前景且周围有背景且周围像素数量小于8
    edge_mask = (edge_mask > 0) & (edge_mask < 4) & (mask == 1)
    
    # 获取边缘点的索引
    indices = torch.nonzero(edge_mask.squeeze())
    pairs = []
    
    for idx in indices:
        p = tuple(idx.tolist())
        for q in [(p[0], p[1]+1), (p[0], p[1]-1), (p[0]+1, p[1]), (p[0]-1, p[1])]:
            if 0 <= q[0] < mask.size(2) and 0 <= q[1] < mask.size(3) and mask[0, 0, q[0], q[1]] != 0:
                pairs.append((p, q))
    
    return pairs

# 计算泊松损失
def poisson_loss(image, v, pairs):
    loss = 0.0
    for p, q in pairs:
        f_p = image[0, :, p[0], p[1]]
        f_q = image[0, :, q[0], q[1]]
        
        # 分离水平和垂直梯度
        v_x_p = v[0, :image.size(1), p[0], p[1]]
        v_x_q = v[0, :image.size(1), q[0], q[1]]
        v_y_p = v[0, image.size(1):, p[0], p[1]]
        v_y_q = v[0, image.size(1):, q[0], q[1]]
        
        # 计算梯度差
        v_x_pq = v_x_p - v_x_q
        v_y_pq = v_y_p - v_y_q
        
        # 计算损失
        for c in range(image.size(1)):
            loss += ((f_p[c] - f_q[c] - v_x_pq[c]) ** 2).sum()
            loss += ((f_p[c] - f_q[c] - v_y_pq[c]) ** 2).sum()
    return loss

# 修改后的损失函数
def cal_laplacian_loss(foreground_img, foreground_mask, blended_img, background_mask):
    device = foreground_img.device

    # 计算梯度
    foreground_gradients = compute_gradient(foreground_img)
    blended_gradients = compute_gradient(blended_img)

    # 应用掩码
    foreground_gradients_masked = foreground_gradients * foreground_mask
    blended_gradients_masked = blended_gradients * background_mask

    # 计算像素对
    pairs = compute_pixel_pairs(background_mask)

    # 计算泊松损失
    loss = poisson_loss(blended_img, blended_gradients_masked, pairs)

    return loss

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            img_input_foreground = gr.Image(label="Foreground Image", tool=None)  # Disable default tool
            polygon_state = gr.State(initialize_polygon())
            btn_close_polygon = gr.Button("Close Polygon")
            img_output_foreground = gr.Image(label="Selected Foreground Area")
        
        with gr.Column():
            img_input_background = gr.Image(label="Background Image", tool=None)  # Disable default tool
            slider_x = gr.Slider(minimum=-100, maximum=100, value=0, label="X Offset")
            slider_y = gr.Slider(minimum=-100, maximum=100, value=0, label="Y Offset")
            img_output_background = gr.Image(label="Background with Overlay")
    
    btn_blend = gr.Button("Blend Images")
    img_output_blended = gr.Image(label="Blended Image")

    # Clear polygon state when new image is uploaded
    def clear_polygon_state(img):
        return img, initialize_polygon()

    img_input_foreground.change(clear_polygon_state, [img_input_foreground], [img_input_foreground, polygon_state])
    img_input_foreground.select(add_point, [img_input_foreground, polygon_state], [img_output_foreground, polygon_state])
    btn_close_polygon.click(close_polygon, [img_input_foreground, polygon_state], [img_output_foreground, polygon_state])
    slider_x.change(update_background, [img_input_background, polygon_state, slider_x, slider_y], img_output_background)
    slider_y.change(update_background, [img_input_background, polygon_state, slider_x, slider_y], img_output_background)
    btn_blend.click(lambda fg, bg, poly, dx, dy: blend_images(fg, bg, poly, dx, dy),
                    [img_input_foreground, img_input_background, polygon_state, slider_x, slider_y],
                    img_output_blended)

# Function to blend images
# Function to blend images
def blend_images(fg_img, bg_img, polygon_state, dx, dy, *args):
    if not polygon_state['closed'] or bg_img is None or fg_img is None:
        return bg_img  # Return original background if conditions are not met

    # Convert images to numpy arrays
    foreground_np = np.array(fg_img)
    background_np = np.array(bg_img)

    # Get polygon points and shift them by dx and dy
    foreground_polygon_points = np.array(polygon_state['points']).astype(np.int64)
    background_polygon_points = foreground_polygon_points + np.array([int(dx), int(dy)]).reshape(1, 2)

    # Create masks from polygon points
    foreground_mask = create_mask_from_points(foreground_polygon_points, foreground_np.shape[0], foreground_np.shape[1])
    background_mask = create_mask_from_points(background_polygon_points, background_np.shape[0], background_np.shape[1])

    # Convert numpy arrays to torch tensors
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fg_img_tensor = torch.from_numpy(foreground_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    bg_img_tensor = torch.from_numpy(background_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    fg_mask_tensor = torch.from_numpy(foreground_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.
    bg_mask_tensor = torch.from_numpy(background_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.

    # Initialize blended image
    blended_img = bg_img_tensor.clone()
    mask_expanded = bg_mask_tensor.bool().expand(-1, 3, -1, -1)
    blended_img[mask_expanded] = blended_img[mask_expanded] * 0.9 + fg_img_tensor[fg_mask_tensor.bool().expand(-1, 3, -1, -1)] * 0.1
    blended_img.requires_grad = True

    # Set up optimizer
    optimizer = torch.optim.Adam([blended_img], lr=1e-3)

    # Optimization loop
    iter_num = 10
    for step in range(iter_num):
        optimizer.zero_grad()
        
        # Compute Laplacian loss
        loss = cal_laplacian_loss(fg_img_tensor, fg_mask_tensor, blended_img, bg_mask_tensor)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        if step % 1 == 0:
            print(f'Optimize step: {step}, Poisson loss: {loss.item()}')

        if step == (iter_num // 2):  # Decrease learning rate at the half step
            optimizer.param_groups[0]['lr'] *= 0.1

    # Convert result back to numpy array
    result = torch.clamp(blended_img.detach(), 0, 1).cpu().permute(0, 2, 3, 1).squeeze().numpy() * 255
    result = result.astype(np.uint8)
    return result

# Launch the interface
demo.launch()
