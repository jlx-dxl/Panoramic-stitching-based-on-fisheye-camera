import numpy as np
import cv2

class Stitcher: #Stitcher类
    #show函数
    def cv_show(self,name,img):
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def put_stitch(self,images):
        (imageA, imageB, imageC) = images
        h = max(imageA.shape[0], imageB.shape[0], imageC.shape[0])
        delta_h=10
        background = np.zeros((h+delta_h, imageA.shape[1] + imageC.shape[1] + imageB.shape[1], 3), dtype=np.uint8)
        background[delta_h:delta_h+imageA.shape[0], 0:imageA.shape[1]] = imageA
        background[0:h, imageA.shape[1]:imageA.shape[1]+imageB.shape[1]] = imageB
        background[delta_h:delta_h+imageC.shape[0], imageA.shape[1]+imageB.shape[1]:background.shape[1]] = imageC
        return background


    def offlinestitch_right(self, images, H):
        (imageA, imageB) = images
        imageB_1 = np.copy(imageB)
        # print("imageB_1.shape:",imageB_1.shape)
        imageA = cv2.copyMakeBorder(imageA, int(0.5*imageA.shape[0]), int(0.5*imageA.shape[0]), 0, int(1.5*imageA.shape[1]), cv2.BORDER_CONSTANT, 0)
        imageB = cv2.copyMakeBorder(imageB, int(0.5*imageB.shape[0]), int(0.5*imageB.shape[0]), 0, int(1.5*imageB.shape[1]), cv2.BORDER_CONSTANT, 0)
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
        return result

    def offlinestitch_left(self, images, H):
        (imageB, imageA) = images
        imageB_1 = np.copy(imageB)
        # print("imageB_1.shape:",imageB_1.shape)
        imageA = cv2.copyMakeBorder(imageA, int(0.5*imageA.shape[0]), int(0.5*imageA.shape[0]), int(1.5*imageA.shape[1]), 0, cv2.BORDER_CONSTANT, 0)
        imageB = cv2.copyMakeBorder(imageB, int(0.5*imageB.shape[0]), int(0.5*imageB.shape[0]), int(1.5*imageB.shape[1]), 0, cv2.BORDER_CONSTANT, 0)
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]),flags=cv2.WARP_INVERSE_MAP)
        return result

    def stitch_all(self,images):
        (imageA, imageB, imageC) = images
        h=max(imageA.shape[0],imageB.shape[0],imageC.shape[0])
        background=np.zeros((h,imageA.shape[1]+imageC.shape[1]-imageB.shape[1],3), dtype=np.uint8)
        # self.cv_show('result1', background)
        background[0:h, 0:imageA.shape[1]]=imageA
        # self.cv_show('result2', background)
        background[0:h, background.shape[1] - imageC.shape[1]:background.shape[1]] = imageC
        # self.cv_show('result3', background)
        background[int(0.5*background.shape[0]-0.5*imageB.shape[0]):int(0.5*background.shape[0]-0.5*imageB.shape[0])+imageB.shape[0],imageA.shape[1]-imageB.shape[1]:imageA.shape[1]]=imageB
        # self.cv_show('result4', background)
        background = background[int(0.275 * background.shape[0]):int(0.75 * background.shape[0]),int(0.175 * background.shape[1]):int(0.85 * background.shape[1])]
        return background

    def stitch_all_add(self,images):
        (imageA, imageB, imageC) = images
        h=max(imageA.shape[0],imageB.shape[0],imageC.shape[0])
        background=np.zeros((h,imageA.shape[1]+imageC.shape[1]-imageB.shape[1],3), dtype=np.uint8)
        # self.cv_show('result1', background)
        background[0:h, 0:imageA.shape[1]] += imageA
        # self.cv_show('result2', background)
        background[0:h, background.shape[1] - imageC.shape[1]:background.shape[1]] += imageC
        # self.cv_show('result3', background)
        background[int(0.5*background.shape[0]-0.5*imageB.shape[0]):int(0.5*background.shape[0]-0.5*imageB.shape[0])+imageB.shape[0],imageA.shape[1]-imageB.shape[1]:imageA.shape[1]] += imageB
        # self.cv_show('result4', background)
        return background

    def stitch_all_1(self,images):
        (imageA, imageB, imageC) = images
        sti = np.zeros((1200,2880,3),dtype = np.uint8)
        sti[35+8:960+35+8,0:960-34] = imageA[0:960,0:960-34]
        sti[8:960+8,960-34:1920-34-145-150] = imageB[0:960, 145:960 - 150]
        sti[0:960, 1920-34-145-150:2880-34-145-150-78] = imageC[0:960, 78:960]
        sti=sti[43:960,0:2880-34-145-150-78]
        return sti


    #拼接函数
    def stitchright(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        #获取输入图片
        (imageA, imageB) = images
        imageB_1 = np.copy(imageB)
        # print("imageB_1.shape:",imageB_1.shape)
        imageA = cv2.copyMakeBorder(imageA, int(0.5*imageA.shape[0]), int(0.5*imageA.shape[0]), 0, int(1.5*imageA.shape[1]), cv2.BORDER_CONSTANT, 0)
        imageB = cv2.copyMakeBorder(imageB, int(0.5*imageB.shape[0]), int(0.5*imageB.shape[0]), 0, int(1.5*imageB.shape[1]), cv2.BORDER_CONSTANT, 0)
        #检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵      
        (matches, H, status) = M
        # 将图片A进行视角变换，result是变换后图片
        # result = cv2.warpPerspective(imageA, H, (1920,1680))
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
        # result = cv2.warpPerspective(imageA, H, (3*imageA.shape[1]+ imageB.shape[1], 3*imageA.shape[0]))
        # print("result_right.shape",result.shape)
        # self.cv_show('result', result)
        # 将图片B传入result图片最左端
        # result[0:imageB_1.shape[0], 0:imageB_1.shape[1]] = imageB_1
        # self.cv_show('result', result)
        # 检测是否需要显示图片匹配
        # if showMatches:
        #     # 生成匹配图片
        #     # vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        #     # 返回结果
        #     return (result)

        # 返回匹配结果
        return result,H

    #拼接函数
    def stitchleft(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        #获取输入图片
        (imageB, imageA) = images
        imageB_1 = np.copy(imageB)
        # print("imageB_1.shape:",imageB_1.shape)
        imageA = cv2.copyMakeBorder(imageA, int(0.5*imageA.shape[0]), int(0.5*imageA.shape[0]), int(1.5*imageA.shape[1]), 0, cv2.BORDER_CONSTANT, 0)
        imageB = cv2.copyMakeBorder(imageB, int(0.5*imageB.shape[0]), int(0.5*imageB.shape[0]), int(1.5*imageB.shape[1]), 0, cv2.BORDER_CONSTANT, 0)
        #检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsB, kpsA, featuresB, featuresA, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵
        (matches, H, status) = M
        # 将图片A进行视角变换，result是变换后图片
        # result = cv2.warpPerspective(imageA, H, (1920,1680))

        result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]),flags=cv2.WARP_INVERSE_MAP)
        # print("result_left.shape",result.shape)
        # result_1=np.copy(result)
        # result_1 = cv2.resize(result_1, None, fx=0.5, fy=0.5)
        # self.cv_show('result1', result)
        # 将图片B传入result图片最右端
        # imageB = cv2.copyMakeBorder(imageB, 0, 0, 500, 200, cv2.BORDER_CONSTANT, 0)
        # result[0:imageB_1.shape[0], result.shape[1]-imageB_1.shape[1]:result.shape[1]] = imageB_1
        # print("imageB.shape",imageB.shape)
        # self.cv_show('result', result)
        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            # vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result)

        # 返回匹配结果
        return result,H

    #SIFT算法（返回特征点集，及对应的特征向量集）
    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.xfeatures2d.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return (kps, features)

    #Brute-Force蛮力匹配算法
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()
  
        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离t的比值小于raio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    # 基于仿射变换的单应性矩阵
    def matchKeypoints_1(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离t的比值小于raio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis
