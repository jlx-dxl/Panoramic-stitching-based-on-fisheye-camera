from Stitcher import Stitcher
import cv2

imageA = cv2.imread('./1.jpg')
imageB = cv2.imread('./2.jpg')

size=640
imageA = cv2.resize(imageA, (size,size))
imageB = cv2.resize(imageB, (size,size))

# 把图片拼接成全景图
stitcher = Stitcher() #创建类的实例化
(result) = stitcher.stitchleft([imageB, imageA], showMatches=True) #调用类中的函数

# 显示所有图片
# stitcher.cv_show("Keypoint Matches", vis)
# cv2.imwrite('Keypoint Matches.jpg',vis)
stitcher.cv_show("Result", result)
# cv2.imwrite('Result.jpg',result)
cv2.waitKey(0)
cv2.destroyAllWindows()