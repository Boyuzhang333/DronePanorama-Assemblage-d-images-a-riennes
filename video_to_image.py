import cv2
import numpy as np
from image_stitching import ImageStitcher

def extract_frame(video_path, save_path, is_water_scene=False):
    cap = cv2.VideoCapture(video_path)
    
    # 设置原始分辨率
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
    
    # 读取两个连续帧
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    
    if ret1 and ret2 and is_water_scene:
        # 创建 ImageStitcher 实例
        stitcher = ImageStitcher(detector_type='sift')  # SIFT 对水面纹理效果较好
        
        # 预处理图像
        frame1_enhanced = stitcher.preprocess_image(frame1)
        frame2_enhanced = stitcher.preprocess_image(frame2)
        
        # 使用 SIFT 特征检测和匹配
        kp1, des1 = stitcher.sift.detectAndCompute(frame1_enhanced, None)
        kp2, des2 = stitcher.sift.detectAndCompute(frame2_enhanced, None)
        
        # 使用 FLANN 匹配器
        matches = stitcher.flann.knnMatch(des1, des2, k=2)
        
        # 应用比率测试筛选好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # 获取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 对齐图像
        height, width = frame1.shape[:2]
        aligned_frame2 = cv2.warpPerspective(frame2, M, (width, height))
        
        # 创建渐变混合
        overlap_width = int(width * 0.15)  # 15% 重叠区域
        gradient = np.zeros((height, width), dtype=np.float32)
        gradient[:, -overlap_width:] = np.linspace(0, 1, overlap_width)
        gradient = np.dstack((gradient, gradient, gradient))
        
        # 混合图像
        blended = cv2.addWeighted(
            frame1 * (1 - gradient), 1,
            aligned_frame2 * gradient, 1,
            0
        )
        
        cv2.imwrite(save_path, blended, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ret1:
        cv2.imwrite(save_path, frame1, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    cap.release()

# 使用示例
video_path = "your_video.mp4"
save_path = "output.jpg"
extract_frame(video_path, save_path, is_water_scene=True) 