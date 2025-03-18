import cv2
import numpy as np
import os
from datetime import datetime
import json

class ImageStitcher:
    def __init__(self):
        self.images = []
        self.image_paths = []
        
    def load_images(self, image_dir):
        """加载图像"""
        print("\n=== 开始加载图像 ===")
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    print(f"加载图像: {filename}, 尺寸: {image.shape}")
                    # 调整图像大小以减少计算量
                    scale = 0.5  # 减小到原来的1/2
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                    self.images.append(image)
                    self.image_paths.append(filename)
        
        print(f"\n总共加载了 {len(self.images)} 张图像")
    
    def preprocess_image(self, image):
        """预处理图像以增强特征点"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 高斯模糊去噪
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        
        return gray
    
    def find_and_match_features(self, img1, img2):
        """使用SIFT和RANSAC查找和匹配特征点"""
        # 预处理图像
        gray1 = self.preprocess_image(img1)
        gray2 = self.preprocess_image(img2)
        
        # 创建SIFT检测器
        sift = cv2.SIFT_create(nfeatures=5000, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        
        # 检测特征点和计算描述符
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, None
            
        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 获取匹配点
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 应用Lowe's比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 30:  # 增加最小匹配点数量
            return None, None, None
            
        # 获取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        return H, mask, good_matches
    
    def blend_images(self, img1, img2, mask, overlap_width=100):
        """使用渐变混合方法混合两张图像"""
        h, w = mask.shape[:2]
        mask = mask.astype(float)
        
        # 创建渐变权重
        for i in range(overlap_width):
            weight = i / overlap_width
            mask[:, i] *= weight
            mask[:, w-1-i] *= (1 - weight)
        
        # 混合图像
        result = img1.copy()
        for c in range(3):
            result[:,:,c] = img1[:,:,c] * (1 - mask) + img2[:,:,c] * mask
            
        return result.astype(np.uint8)
    
    def stitch_pair(self, img1, img2):
        """拼接两张图像"""
        # 找到变换矩阵
        H, mask, matches = self.find_and_match_features(img1, img2)
        
        if H is None:
            return False, None
            
        # 计算变换后的图像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
        pts2 = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts2, np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)))
        
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1,0,t[0]], [0,1,t[1]], [0,0,1]])
        
        # 执行透视变换
        warped1 = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
        
        # 创建掩码
        mask = np.zeros((ymax-ymin, xmax-xmin), dtype=np.float32)
        mask[t[1]:h2+t[1], t[0]:w2+t[0]] = 1
        
        # 将第二张图像放入正确位置
        warped2 = np.zeros_like(warped1)
        warped2[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
        
        # 混合图像
        result = self.blend_images(warped1, warped2, mask)
        
        return True, result
    
    def stitch_all(self):
        """拼接所有图像"""
        if len(self.images) < 2:
            print("需要至少两张图像才能进行拼接")
            return None
            
        print("\n=== 开始拼接过程 ===\n")
        
        # 从第一张图像开始
        result = self.images[0]
        cv2.imwrite("stitched_0.jpg", result)
        
        # 逐张拼接
        for i in range(1, len(self.images)):
            print(f"\n尝试拼接 {self.image_paths[i-1]} 和 {self.image_paths[i]}")
            success, stitched = self.stitch_pair(result, self.images[i])
            
            if success:
                result = stitched
                print(f"成功拼接 {self.image_paths[i]}")
                # 保存中间结果
                cv2.imwrite(f"stitched_{i}.jpg", result)
            else:
                print(f"拼接失败，跳过图像 {self.image_paths[i]}")
        
        # 保存最终结果
        if result is not None:
            # 调整最终图像大小
            scale = 2.0  # 因为之前缩小了0.5
            result = cv2.resize(result, None, fx=scale, fy=scale)
            cv2.imwrite("stitched_result.jpg", result)
            print("\n拼接完成，结果已保存为 stitched_result.jpg")
            return result
        else:
            print("\n拼接失败")
            return None

def main():
    # 创建拼接器实例
    stitcher = ImageStitcher()
    
    # 加载图像
    stitcher.load_images("images")
    
    # 执行拼接
    result = stitcher.stitch_all()

if __name__ == "__main__":
    main() 