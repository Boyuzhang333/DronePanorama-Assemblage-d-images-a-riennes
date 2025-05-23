import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional
import re
import gc

class ImageStitcher:
    def __init__(self, detector_type='sift'):
        """
        初始化图像拼接器
        detector_type: 特征检测器类型 ('sift', 'orb', 'akaze')
        """
        self.detector_type = detector_type.lower()
        
        # 初始化SIFT检测器
        self.sift = cv2.SIFT_create(
            nfeatures=10000,
            nOctaveLayers=5,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        
        # 初始化ORB检测器
        self.orb = cv2.ORB_create(
            nfeatures=10000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=3,
            patchSize=41,
            fastThreshold=20
        )
        
        # 初始化AKAZE检测器
        self.akaze = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.0008,
            nOctaves=5,
            nOctaveLayers=5,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
        
        # 初始化匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def load_images(self, image_folder: str) -> List[np.ndarray]:
        """从指定文件夹加载所有图片"""
        images = []
        image_paths = []  # 保持返回两个值
        
        # 确保文件夹存在
        if not os.path.exists(image_folder):
            print(f"错误：找不到文件夹 {image_folder}")
            return images, image_paths
            
        # 获取所有图片文件
        for filename in sorted(os.listdir(image_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"已加载图片: {filename}")
                    images.append(img)
                    image_paths.append(img_path)
                else:
                    print(f"警告：无法加载图片 {filename}")
                    
        return images, image_paths

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：增强对比度和细节"""
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道进行CLAHE处理
        l = clahe.apply(l)
        
        # 合并通道
        lab = cv2.merge((l,a,b))
        
        # 转换回BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 轻微锐化
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def find_and_match_features(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.7) -> List[cv2.DMatch]:
        """查找和匹配两张图片之间的特征点"""
        
        # 检测特征点和描述符
        if self.detector_type == 'sift':
            kp1, des1 = self.sift.detectAndCompute(img1, None)
            kp2, des2 = self.sift.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None:
                return []
            
            matches = self.flann.knnMatch(des1, des2, k=2)
            good_matches = []
            
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
            
            print(f"SIFT检测到特征点: 图片1={len(kp1)}, 图片2={len(kp2)}")
            print(f"SIFT匹配点数量: {len(good_matches)}")
            
        elif self.detector_type == 'orb':
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None:
                return []
            
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            good_matches = matches[:1000]
            print(f"ORB检测到特征点: 图片1={len(kp1)}, 图片2={len(kp2)}")
            print(f"ORB匹配点数量: {len(good_matches)}")
            
        else:  # akaze
            kp1, des1 = self.akaze.detectAndCompute(img1, None)
            kp2, des2 = self.akaze.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None:
                return []
            
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            good_matches = matches[:1000]
            print(f"AKAZE检测到特征点: 图片1={len(kp1)}, 图片2={len(kp2)}")
            print(f"AKAZE匹配点数量: {len(good_matches)}")

        if len(good_matches) < 4:
            print("匹配点数量不足")
            return []

        return good_matches

    def compute_homography(self, matches: List[cv2.DMatch], kp1, kp2) -> Optional[np.ndarray]:
        """计算单应性矩阵"""
        if not matches:
            return None
            
        # 从关键点中获取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # 使用RANSAC方法计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H if H is not None else None

    def color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        将目标图像的颜色风格转换为源图像的风格
        """
        # 转换到 L*a*b* 色彩空间
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 分别计算均值和标准差
        source_mean, source_std = [], []
        target_mean, target_std = [], []
        
        for i in range(3):
            source_mean.append(np.mean(source_lab[:,:,i]))
            source_std.append(np.std(source_lab[:,:,i]))
            target_mean.append(np.mean(target_lab[:,:,i]))
            target_std.append(np.std(target_lab[:,:,i]))
        
        # 对每个通道进行颜色转换
        for i in range(3):
            target_lab[:,:,i] = ((target_lab[:,:,i] - target_mean[i]) * 
                                (source_std[i] / target_std[i])) + source_mean[i]
        
        # 确保值在合理范围内
        target_lab = np.clip(target_lab, 0, 255)
        
        # 转换回BGR
        result = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def multi_band_blend(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, 
                        num_levels: int = 4) -> np.ndarray:
        """
        多频段图像融合
        """
        # 创建高斯金字塔
        img1_pyr = [img1.astype(np.float32)]
        img2_pyr = [img2.astype(np.float32)]
        mask_pyr = [mask.astype(np.float32)]
        
        for i in range(num_levels - 1):
            img1_pyr.append(cv2.pyrDown(img1_pyr[-1]))
            img2_pyr.append(cv2.pyrDown(img2_pyr[-1]))
            mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))
        
        # 创建拉普拉斯金字塔
        lap1_pyr = [img1_pyr[num_levels-1]]
        lap2_pyr = [img2_pyr[num_levels-1]]
        
        for i in range(num_levels-1, 0, -1):
            size = (img1_pyr[i-1].shape[1], img1_pyr[i-1].shape[0])
            expanded = cv2.pyrUp(img1_pyr[i], dstsize=size)
            lap1_pyr.append(img1_pyr[i-1] - expanded)
            expanded = cv2.pyrUp(img2_pyr[i], dstsize=size)
            lap2_pyr.append(img2_pyr[i-1] - expanded)
        
        # 融合金字塔
        blend_pyr = []
        for i in range(num_levels):
            blend = lap1_pyr[i] * mask_pyr[i] + lap2_pyr[i] * (1 - mask_pyr[i])
            blend_pyr.append(blend)
        
        # 重建图像
        result = blend_pyr[0]
        for i in range(1, num_levels):
            size = (blend_pyr[i].shape[1], blend_pyr[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size) + blend_pyr[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def create_panorama(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """创建全景图"""
        # 获取图像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算变换后的四个角点
        corners = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
        corners_transformed = cv2.perspectiveTransform(corners, H)
        
        # 计算输出图像的尺寸，确保包含所有图像区域，并增加30%的边距
        xmin = int(min(corners_transformed[:,:,0].min(), 0) * 1.3)
        ymin = int(min(corners_transformed[:,:,1].min(), 0) * 1.3)
        xmax = int(max(corners_transformed[:,:,0].max(), w2) * 1.3)
        ymax = int(max(corners_transformed[:,:,1].max(), h2) * 1.3)
        
        # 调整变换矩阵
        offset = [-xmin, -ymin]
        H_adjusted = np.array([[1,0,offset[0]], [0,1,offset[1]], [0,0,1]]) @ H
        
        # 创建输出图像
        output_size = (xmax-xmin, ymax-ymin)
        result = cv2.warpPerspective(img1, H_adjusted, output_size)
        
        # 将第二张图片复制到结果中
        y_offset = int(offset[1])
        x_offset = int(offset[0])
        y_end = min(y_offset + h2, result.shape[0])
        x_end = min(x_offset + w2, result.shape[1])
        
        for y in range(y_offset, y_end):
            for x in range(x_offset, x_end):
                if 0 <= y-y_offset < h2 and 0 <= x-x_offset < w2:
                    result[y,x] = img2[y-y_offset,x-x_offset]
        
        # 裁剪黑边
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            result = result[y:y+h, x:x+w]
        
        return result

    def crop_black_edges(self, image: np.ndarray) -> np.ndarray:
        """裁剪图像边缘的黑色区域"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用阈值找到非黑色区域
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 找到轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # 裁剪图像
            return image[y:y+h, x:x+w]
        
        return image

    def stitch_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        简化的图像拼接方法，保持原图效果
        """
        if len(images) < 2:
            return images[0] if images else None
        
        result = images[0]
        
        for i in range(1, len(images)):
            # 检测特征点和描述符
            if self.detector_type == 'sift':
                kp1, des1 = self.sift.detectAndCompute(result, None)
                kp2, des2 = self.sift.detectAndCompute(images[i], None)
                
                if des1 is None or des2 is None:
                    continue
                    
                # 特征匹配
                matches = self.flann.knnMatch(des1, des2, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
            else:
                # 对于其他检测器的处理保持不变
                good_matches = self.find_and_match_features(result, images[i], threshold=0.8)
            
            if len(good_matches) < 4:
                continue
            
            # 计算变换矩阵
            H = self.compute_homography(good_matches, kp1, kp2)
            if H is None:
                continue
            
            # 创建全景图
            result = self.create_panorama(result, images[i], H)
        
        return result

def get_test_number(input_path):
    """从输入路径中提取测试序号"""
    # 从路径中提取 'imageTestX' 中的 X
    match = re.search(r'imageTest(\d+)', input_path)
    if match:
        return match.group(1)
    return "1"  # 默认返回1

def select_detector():
    """让用户选择特征检测器"""
    print("\n可用的特征检测器：")
    print("1. SIFT (尺度不变特征变换)")
    print("2. ORB (定向FAST和旋转BRIEF)")
    print("3. AKAZE (加速KAZE)")
    print("4. 全部都试一下")
    
    while True:
        choice = input("\n请选择特征检测器 (1-4): ").strip()
        if choice == '1':
            return ['sift']
        elif choice == '2':
            return ['orb']
        elif choice == '3':
            return ['akaze']
        elif choice == '4':
            return ['sift', 'orb', 'akaze']
        else:
            print("无效的选择，请重试")

def process_black_borders(image: np.ndarray, method='crop') -> np.ndarray:
    """
    处理图像中的黑色边框
    
    Args:
        image: 输入图像
        method: 处理方法，'crop'表示裁剪，'scale'表示缩放填充
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 找到非黑色区域
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 找到非零区域的边界框
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    # 获取最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    if method == 'crop':
        # 直接裁剪
        return image[y:y+h, x:x+w]
    else:  # method == 'scale'
        # 先裁剪有效区域
        cropped = image[y:y+h, x:x+w]
        # 计算原始图片尺寸
        orig_h, orig_w = image.shape[:2]
        # 计算缩放比例
        scale_w = orig_w / w
        scale_h = orig_h / h
        scale = max(scale_w, scale_h)
        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 缩放图片
        result = cv2.resize(cropped, (new_w, new_h))
        # 如果缩放后的图片大于原始尺寸，裁剪到原始尺寸
        if new_w > orig_w or new_h > orig_h:
            start_x = (new_w - orig_w) // 2 if new_w > orig_w else 0
            start_y = (new_h - orig_h) // 2 if new_h > orig_h else 0
            result = result[start_y:start_y+orig_h, start_x:start_x+orig_w]
        return result

def main():
    """主函数"""
    # 获取用户输入的路径
    input_folder = input("请输入子图片所在文件夹路径（例如：images/imageTest1）: ").strip()
    if not os.path.exists(input_folder):
        print(f"错误：找不到文件夹 {input_folder}")
        return
        
    # 根据输入路径生成输出路径
    test_number = get_test_number(input_folder)
    output_folder = f"imageResultat{test_number}"
    
    # 确保输出文件夹存在
    Path(output_folder).mkdir(exist_ok=True)
    
    print(f"\n输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    
    # 让用户选择特征检测器
    detectors = select_detector()
    
    # 让用户选择黑边处理方式
    print("\n请选择黑边处理方式：")
    print("1. 直接裁剪黑边")
    print("2. 缩放填充")
    print("3. 保持原样不处理")
    while True:
        border_choice = input("请选择处理方式 (1-3): ").strip()
        if border_choice in ['1', '2', '3']:
            break
        print("无效的选择，请重试")
    
    border_method = None
    if border_choice == '1':
        border_method = 'crop'
    elif border_choice == '2':
        border_method = 'scale'
    
    # 使用选定的检测器进行拼接
    for detector in detectors:
        print(f"\n使用 {detector.upper()} 检测器进行拼接...")
        stitcher = ImageStitcher(detector_type=detector)
        
        # 加载图片
        images, image_paths = stitcher.load_images(input_folder)
        if not images:
            print("错误：没有找到有效的图片文件")
            continue
            
        print(f"\n成功加载 {len(images)} 张子图片")
        print("图片加载顺序：")
        for path in image_paths:
            print(f"  - {os.path.basename(path)}")
        
        # 开始拼接
        result = stitcher.stitch_images(images)
        
        if result is not None:
            # 处理黑边
            if border_method:
                print(f"\n正在处理黑边...")
                result = process_black_borders(result, border_method)
            
            # 生成输出文件名
            output_path = os.path.join(output_folder, f'stitched_result_{detector}.jpg')
            print(f"\n保存 {detector.upper()} 结果到 {output_path}...")
            cv2.imwrite(output_path, result)
            print("保存完成")
        else:
            print(f"\n{detector.upper()} 拼接失败，无法生成结果")

if __name__ == "__main__":
    main() 