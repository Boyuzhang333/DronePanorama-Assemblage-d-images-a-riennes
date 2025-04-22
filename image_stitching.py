import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class ImageStitcher:
    def __init__(self):
        # 初始化SIFT特征检测器
        self.sift = cv2.SIFT_create(
            nfeatures=10000,  # 增加特征点数量
            nOctaveLayers=5,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        # 初始化FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def load_images(self, image_folder: str) -> List[np.ndarray]:
        """加载文件夹中的所有图片"""
        images = []
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"Loaded image: {filename}")
                    images.append(img)
        return images

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

    def find_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """查找和匹配两张图片之间的特征点"""
        # 检测特征点和描述符
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return None, None, []

        # 使用FLANN进行特征匹配
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # 应用Lowe's ratio测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        print(f"Found {len(good_matches)} good matches")
        
        if len(good_matches) < 10:
            return None, None, []

        # 获取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return src_pts, dst_pts, good_matches

    def compute_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """计算单应性矩阵"""
        if src_pts is None or dst_pts is None:
            return None
            
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H if H is not None else None

    def color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """将源图像的颜色特征转移到目标图像"""
        # 转换到LAB色彩空间
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 分别计算均值和标准差
        source_mean = np.mean(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))

        # 调整目标图像
        target_lab = (target_lab - target_mean) * (source_std / target_std) + source_mean

        # 确保值在有效范围内
        target_lab = np.clip(target_lab, 0, 255)

        # 转换回BGR
        result = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def multi_band_blend(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, num_levels: int = 4) -> np.ndarray:
        """多频段融合"""
        # 创建高斯金字塔
        img1_pyr = [img1.astype(np.float32)]
        img2_pyr = [img2.astype(np.float32)]
        mask_pyr = [mask.astype(np.float32)]
        
        for _ in range(num_levels):
            img1_pyr.append(cv2.pyrDown(img1_pyr[-1]))
            img2_pyr.append(cv2.pyrDown(img2_pyr[-1]))
            mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))
        
        # 在最低分辨率层进行融合
        result_pyr = [img1_pyr[-1] * mask_pyr[-1] + img2_pyr[-1] * (1 - mask_pyr[-1])]
        
        # 逐层向上融合
        for i in range(num_levels-1, -1, -1):
            size = (img1_pyr[i].shape[1], img1_pyr[i].shape[0])
            up_sampled = cv2.pyrUp(result_pyr[0], dstsize=size)
            result_pyr.insert(0, img1_pyr[i] * mask_pyr[i] + 
                               img2_pyr[i] * (1 - mask_pyr[i]) +
                               up_sampled)
        
        return np.clip(result_pyr[0], 0, 255).astype(np.uint8)

    def create_panorama(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """创建全景图"""
        # 获取图像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算变换后的四个角点
        corners = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
        corners_transformed = cv2.perspectiveTransform(corners, H)
        
        # 计算输出图像的尺寸
        xmin = min(corners_transformed[:,:,0].min(), 0)
        ymin = min(corners_transformed[:,:,1].min(), 0)
        xmax = max(corners_transformed[:,:,0].max(), w2)
        ymax = max(corners_transformed[:,:,1].max(), h2)
        
        # 调整变换矩阵以消除黑边
        offset = [-xmin, -ymin]
        H_adjusted = np.array([[1,0,offset[0]], [0,1,offset[1]], [0,0,1]]) @ H
        
        # 创建输出图像
        output_size = (int(xmax-xmin), int(ymax-ymin))
        result = cv2.warpPerspective(img1, H_adjusted, output_size)
        
        # 将第二张图片复制到结果中
        y_offset = int(offset[1])
        x_offset = int(offset[0])
        y_end = min(y_offset + h2, result.shape[0])
        x_end = min(x_offset + w2, result.shape[1])
        
        # 创建简单的alpha混合区域
        overlap_width = 100  # 重叠区域宽度
        for y in range(y_offset, y_end):
            for x in range(x_offset, x_end):
                if x - x_offset < overlap_width:
                    # 在重叠区域使用渐变权重
                    alpha = (x - x_offset) / overlap_width
                    if 0 <= y-y_offset < h2 and 0 <= x-x_offset < w2:
                        result[y,x] = (1-alpha) * result[y,x] + alpha * img2[y-y_offset,x-x_offset]
                else:
                    if 0 <= y-y_offset < h2 and 0 <= x-x_offset < w2:
                        result[y,x] = img2[y-y_offset,x-x_offset]
        
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

    def stitch_images(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """拼接多张图片"""
        if not images:
            return None
        
        # 从第一张图片开始
        result = images[0]
        
        # 逐个拼接后续图片
        for i in range(1, len(images)):
            print(f"\nProcessing image pair {i}/{len(images)-1}")
            
            # 查找和匹配特征点
            src_pts, dst_pts, matches = self.find_and_match_features(result, images[i])
            
            if src_pts is None or len(matches) < 10:
                print(f"Not enough matches found for image {i}, skipping...")
                continue
            
            # 计算单应性矩阵
            H = self.compute_homography(src_pts, dst_pts)
            
            if H is None:
                print(f"Could not compute homography for image {i}, skipping...")
                continue
            
            # 创建全景图
            result = self.create_panorama(result, images[i], H)
            print(f"Successfully stitched image {i}")
        
        return result

def main():
    # 创建 ImageStitcher 实例
    stitcher = ImageStitcher()
    
    # 加载图片
    image_folder = "images"  # 替换为你的图片文件夹路径
    images = stitcher.load_images(image_folder)
    
    if not images:
        print("No images found in the specified folder")
        return
    
    print(f"Loaded {len(images)} images")
    
    # 拼接图片
    result = stitcher.stitch_images(images)
    
    if result is not None:
        # 保存结果
        cv2.imwrite("stitched_result.jpg", result)
        print("Stitching completed and saved as 'stitched_result.jpg'")
    else:
        print("Failed to stitch images")

if __name__ == "__main__":
    main() 