import cv2
import numpy as np
import os
from datetime import datetime
import json

def create_base_image(width, height):
    """创建基础图像，包含丰富的特征点"""
    # 创建基础图像（模拟水面）
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array([100, 150, 200], dtype=np.uint8)
    
    # 添加网格特征
    grid_size = 30  # 减小网格大小
    for i in range(0, width, grid_size):
        cv2.line(image, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, grid_size):
        cv2.line(image, (0, i), (width, i), (200, 200, 200), 1)
    
    # 添加随机圆形特征
    for _ in range(300):  # 增加特征点数量
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(3, 10)  # 减小圆的大小
        color = tuple(map(int, np.random.randint(150, 256, 3).tolist()))
        cv2.circle(image, (x, y), radius, color, -1)
    
    # 添加随机矩形特征
    for _ in range(200):  # 增加特征点数量
        x = np.random.randint(0, width-30)
        y = np.random.randint(0, height-30)
        w = np.random.randint(10, 30)  # 减小矩形大小
        h = np.random.randint(10, 30)
        color = tuple(map(int, np.random.randint(150, 256, 3).tolist()))
        cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
    
    # 添加随机线条
    for _ in range(100):  # 增加特征点数量
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(max(0, x1-50), min(width, x1+50))  # 减小线条长度
        y2 = np.random.randint(max(0, y1-50), min(height, y1+50))
        color = tuple(map(int, np.random.randint(150, 256, 3).tolist()))
        cv2.line(image, (x1, y1), (x2, y2), color, 1)  # 减小线条宽度
    
    return image

def generate_test_data(num_images=8):  # 减少图像数量
    """生成测试数据"""
    # 创建输出目录
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成GPS数据
    gps_data = {}
    
    # 基础参数
    base_lat = 43.5500  # 基础纬度
    base_lon = 7.0167   # 基础经度
    lat_step = 0.0001   # 纬度步长
    lon_step = 0.0001   # 经度步长
    
    # 创建大图
    big_width = 1600    # 减小大图尺寸
    big_height = 1200
    big_image = create_base_image(big_width, big_height)
    
    # 从大图中裁剪出重叠的子图
    window_width = 800   # 子图宽度
    window_height = 600  # 子图高度
    overlap = 0.7        # 增加重叠比例
    
    step_x = int(window_width * (1 - overlap))
    step_y = int(window_height * (1 - overlap))
    
    image_count = 0
    for y in range(0, big_height - window_height + 1, step_y):
        for x in range(0, big_width - window_width + 1, step_x):
            if image_count >= num_images:
                break
                
            # 裁剪子图
            window = big_image[y:y+window_height, x:x+window_width].copy()
            
            # 添加一些随机变化
            # 亮度变化
            brightness = np.random.uniform(0.95, 1.05)  # 减小亮度变化范围
            window = cv2.convertScaleAbs(window, alpha=brightness, beta=0)
            
            # 轻微的旋转
            angle = np.random.uniform(-0.5, 0.5)  # 减小旋转角度
            center = (window_width//2, window_height//2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            window = cv2.warpAffine(window, M, (window_width, window_height))
            
            # 保存图像
            filename = f"image_{image_count:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), window)
            
            # 生成GPS数据
            gps_data[filename] = {
                "latitude": base_lat + y * lat_step,
                "longitude": base_lon + x * lon_step
            }
            
            image_count += 1
            if image_count >= num_images:
                break
    
    # 保存GPS数据
    with open("gps_data.json", "w") as f:
        json.dump(gps_data, f, indent=4)
    
    print(f"已生成 {image_count} 张测试图像和对应的GPS数据")

if __name__ == "__main__":
    generate_test_data() 