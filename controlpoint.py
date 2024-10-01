import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None



# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换


def vertical_vector(arr):  
    # 显式地创建一个浮点数数组  
    b = np.array([0.0, 0.0])  
    b[0] = -arr[1]  
    b[1] = arr[0]  
    return b


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    deformed_image = np.zeros_like(image)   
    warped_image = np.array(image)
    h, w = image.shape[:2]  # 获取图像的高度和宽度  
    x_range = np.linspace(0, w-1, 1)  # 在x方向上生成网格点  
    y_range = np.linspace(0, h-1, 1)  # 在y方向上生成网格点  
    for y in range(h):    
        for x in range(w):    
            grid_point = np.array([x, y])  # 当前网格点的坐标  
            new_point = np.array([0,0])
            p_average = np.array([0,0])    #当前点对应的平均目标点
            q_average = np.array([0,0])    #当前点对应的平均控制点
            u_s=w_sum_p = 0.0
            for i in range(len(source_pts)):
                w_now_p = 1/(np.linalg.norm(grid_point-source_pts[i])**2+eps)
                w_sum_p = w_sum_p+w_now_p
                p_average = p_average+w_now_p*source_pts[i]
                q_average = q_average+w_now_p*target_pts[i]
            p_average = p_average/w_sum_p       #计算p*
            q_average = q_average/w_sum_p       #计算q*
            for i in range(len(source_pts)):
                u_s = u_s +  1/(np.linalg.norm(grid_point-source_pts[i])**2+eps)*np.dot(source_pts[i]-p_average,source_pts[i]-p_average)  #计算us
            
            for i in range(len(source_pts)):
                w_i = 1/(np.linalg.norm(grid_point-source_pts[i])**2+eps)
                A_i1 = np.array([source_pts[i]-p_average,-1*vertical_vector(source_pts[i]-p_average)])
                A_i2 = np.array([grid_point-p_average,-vertical_vector(grid_point-p_average)])
                A_i = w_i*A_i1@(A_i2).T
                new_point = new_point+(1/u_s)*np.dot((target_pts[i]-q_average),A_i)

            new_point = new_point+q_average
            clipped_x = np.clip(new_point[0], 0, w-1)  
            new_x = int(clipped_x) if not np.isnan(clipped_x) else 0
            clipped_y = np.clip(new_point[1], 0, h-1)  
            new_y = int(clipped_y) if not np.isnan(clipped_y) else 0
            deformed_image[new_y, new_x] = image[y, x]    

    # 初始化变形后的图像，与原图同样大小但全为0    

    ### FILL: 基于MLS or RBF 实现 image warping

    return deformed_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()