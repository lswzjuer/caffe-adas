import os
import numpy as np
import cv2

from skimage import measure

class laneProcess():
    def __init__(self, savepath, ratio):
        """Give a specific path to save debug images
        """
        #print("finish init")
        self._all_ext_points = []
        self._regionNum = 0
        self._regionList = []
        self._regionLabels = []
        self._laneList = []
        self._laneTypes = ["l1", "r1", "l2", "r2", "mid"]
        
        self._path = savepath
        self._ratio = ratio   # scale parameters (srcImg.height/predImg.height)
        
    def readImage(self, imagepath):
        """Read image by image path
        """
        assert(os.path.exists(imagepath))
        image = cv2.imread(imagepath)
        return image
    
    def parseImglist(self, imagelist):
        """parse image name file to get image list
        """
        assert(os.path.exists(imagelist))
        #print("imagelist: ", imagelist)
        with open(imagelist, 'r') as f:
            lines = f.readlines()
        return lines 
    
    def colorMap(self): 
        """Different for different lanes
        """
        # l1: red   r1: blue   l2: green  r2: yellow  mid: purple
        color_map = {"l1":(0, 0, 255), "r1":(255, 0, 0), "l2":(0, 255, 0), "r2":(0, 255, 255), "mid":(255, 48, 155)}
        return color_map

    def findMostPixelValue(self, image, regionCoords):
        """Traverse pixels in the specific region by coords it provided, choose the majority label as the lane
        type of this region.
        params:
            image: The NN predicted image
            regionCoords: The coords of pixels in one specific region.
        return:
            laneType: Lane type the given region represents in image.
        """
        N, _ = regionCoords.shape
        l1_cnt = 0
        r1_cnt = 0
        l2_cnt = 0
        r2_cnt = 0
        m_cnt = 0
        for p_index in range(0, N):
            p_x, p_y = regionCoords[p_index, :]
            pixelValue = image[p_x, p_y]
            if pixelValue == 1:
                l1_cnt += 1
            elif pixelValue == 2:
                r1_cnt += 1
            elif pixelValue == 3:
                l2_cnt += 1
            elif pixelValue == 4:
                r2_cnt += 1
            elif pixelValue == 5:
                m_cnt += 1
            else:
                pass
        cnts = np.array([l1_cnt, r1_cnt, l2_cnt, r2_cnt, m_cnt])
        #print("cnts: ", cnts)
        laneType = self._laneTypes[np.argmax(cnts)]
        return laneType
        
    
    def filterRegion(self, predImg, regionImg, areaThreshold = 20, lengthThreshold = 8):
        """Remove those region whose area is lower than areaThreshold or maxLength is short than lengthThreshold.
        params:
            predImg: The NN predicted image
            regionImg: The image which has been labeled again by different connected region
        return:
            regionImg: Cleaned regionImg
            regionList: The region labels list which has been kept
            laneList: The lane types list. The elements num in laneList should be same with that in regionList.
                    E.g: ['l1', 'r1', ...]
        """
        prop = measure.regionprops(regionImg)
        
        regionNum = len(np.unique(regionImg))
        regionList = []
        laneList = []
        
        for l_index in range(1, regionNum):
            if prop[l_index - 1].area < areaThreshold:
                regionImg[np.where(regionImg == l_index)] = 0
            elif prop[l_index - 1].major_axis_length < lengthThreshold:
                regionImg[np.where(regionImg == l_index)] = 0
            else:
                laneType = self.findMostPixelValue(predImg, prop[l_index - 1].coords)
                laneList.append(laneType)
                regionList.append(l_index)
                #print("l_index, laneType: ", l_index, laneType, len(prop[l_index - 1].coords))
        
        #print("regionList: ", regionList, regionNum)
        return regionImg, regionList, laneList
    
    def getRegionLabel(self, predImg):
        """Use skimage.measure to find the connecte regions in image predicted by NN.
        params:
            predImg: The NN predicted image
        return:
            Change this variable values. _regionNum, _regionList, _regionLabels, _laneList
        """
        assert(predImg is not None)
        
        ## convert image to binary
        #image[np.where(image != 0)] = 1
        
        regionImg = measure.label(predImg, connectivity=2)

        labels, regionList, laneList = self.filterRegion(predImg, regionImg, 100/self._ratio, 40/self._ratio)
        
        self._regionNum = len(np.unique(labels)) - 1  # ignore background label 0
        self._regionList = regionList
        self._regionLabels = labels
        self._laneList = laneList
        #print("regionNum: ", self._regionNum)
        
        assert(len(self._regionList) == self._regionNum)
        assert(len(self._regionList) == len(self._laneList))
        
    def getLanePoints(self):
        """Get the representative points for different lanes. For each lane, select the mean pixel in each row of self._regionLabels
        as the lane points. Here 
        
        """
        laneNum = self._regionNum
        laneImg = self._regionLabels
        height, width = laneImg.shape
        
        allLanePoints = []
        for l_index in range(0, laneNum):  ## laneNum has remove background
            laneLabel = self._regionList[l_index]
            #print("index, label: ", l_index, laneLabel)
            sigleLanePoints = []
            for r_index in range(0, height):
                row_data = laneImg[r_index, :]
                
                points_x = np.where(row_data == laneLabel)
                
                if len(points_x[0]) == 0:
                    continue
                points_x = int(np.mean(points_x)) * self._ratio
                points_y = int(r_index) * self._ratio
                sigleLanePoints.append([points_x, points_y])
            
            #print("sigleChannel: ", len(sigleLanePoints))
            allLanePoints.append(sigleLanePoints)
        self._allLanePoints = np.array(allLanePoints)
        #print("allLanePoints: ", self._allLanePoints.shape)
    
    def getExtPoints(self, height):  
        """Get the extension points for all lanes.
        param:
            height: Used to calculate the extension points in the bottom of image.
        """
        hNum = height
        
        laneNum = self._regionNum
        points = self._allLanePoints
        
        all_ext_points = []
        
        for c_index in range(0, laneNum):
            linePoints = points[c_index]
            
            pointsNum = len(linePoints)
            
            pointsArray = np.array(linePoints)  # x, y
            x = pointsArray[:, 0]
            y = pointsArray[:, 1]
            
            z1 = np.polyfit(y, x, 1)   ## fit a line
            lineFunc = np.poly1d(z1)
            #print("lineFunc: ", lineFunc)
            
            ext_y = hNum
            ext_x = int(lineFunc(ext_y))
            
            all_ext_points.append([ext_x, ext_y])
                  
        self._all_ext_points = all_ext_points

    def getSigleLaneExtPoints(self, laneName, filterType):
        """Get ext points of lane L1, R1 and M by filtering the coords. 
        Here, self._laneList should have more than one L1(R1/M)
        """
        all_ext_points = np.array(self._all_ext_points)
        laneList = np.array(self._laneList)
        
        target_index = np.where(laneList == laneName)[0]
        target_points_x = all_ext_points[target_index][:, 0]
        #print("target_index: ", target_index, target_points_x) 
       
        coord = []
        if filterType == "larger":
            maxIndex = np.argmax(target_points_x)
            maxIndex = target_index[maxIndex]
            coord = all_ext_points[maxIndex]
        if filterType == "small":
            minIndex = np.argmin(target_points_x)
            minIndex = target_index[minIndex]
            coord = all_ext_points[minIndex]
        
        #print("coord: ", coord)
        return coord
    
    def getTargetLaneExtPoints(self):
        all_ext_points = np.array(self._all_ext_points)
        laneList = np.array(self._laneList)
        
        ## l1 lane
        all_index = np.where(laneList == "l1")[0]
        if len(all_index) > 0:
            l1_coord = self.getSigleLaneExtPoints("l1", "larger")
        else:
            l1_coord = []
        
        # r1 lane
        all_index = np.where(laneList == "r1")[0]
        if len(all_index) > 0:
            r1_coord = self.getSigleLaneExtPoints("r1", "small")
        else:
            r1_coord = []
        
        # mid lane
        all_index = np.where(laneList == "mid")[0]
        if len(all_index) > 0:
            mid_coord = self.getSigleLaneExtPoints("mid", "small")
        else:
            mid_coord = []
            
        targetLane = [l1_coord, r1_coord, mid_coord]
        #print("targetLane: ", targetLane)
        return targetLane
        
    
    def drawPredPoints(self, srcImg):
        """draw raw predict points on source image
        params: 
            points: The uncleaned  predicted points. np.ndarray [C], C represents the line types, each element in C-dim is a tuple. Here C is \
                equal to self.initLaneNum
        """

        points = self._allLanePoints
        #print("points: ", points.shape)
        color_map = self.colorMap()
        decodeImage = srcImg.copy()
        for c_index in range(0, self._regionNum):  # dont not handle background
            sigleChannel = points[c_index]
            pNum = len(sigleChannel)
            #print("c_index--pNum", c_index, pNum)
            for p_index in range(0, pNum):
                x, y = sigleChannel[p_index]    
                laneType = self._laneList[c_index]
                cv2.circle(decodeImage, (int(x), int(y)), 2, color_map[laneType], 2)
                
                """
                if not len(self._all_ext_points) == 0:
                    ext_x, ext_y = self._all_ext_points[c_index]
                    cv2.circle(decodeImage, (ext_x, ext_y), 4, (255, 245, 152), 4)
                """
        cv2.imwrite(self._path + self._imgName + "_predPoints.jpg", decodeImage) 

    def drawTargetlaneExtPoints(self, srcImg, targetLane):
        decodeImage = srcImg.copy()
        laneMap = ["l1", "r1", "mid"]
        color_map = self.colorMap()
        for i in range(0, len(targetLane)):
            if len(targetLane[i]) > 0:
                x, y = targetLane[i]
                cv2.circle(decodeImage, (int(x), int(y)), 10, color_map[laneMap[i]], 10)
        cv2.imwrite(self._path + self._imgName + "_targetPoints.jpg", decodeImage)
    
    def drawExtPoints(self, srcImg, all_ext_points, targetPoints): 
        """draw the lane line in source image.          
        """
        hNum, wNum, _ = srcImg.shape
        decodeImage = srcImg.copy()
        color_map = self.colorMap()
        
        laneNum = self._regionNum
        points = self._allLanePoints
        
        for c_index in range(0, laneNum):
            linePoints = points[c_index]
            
            pointsNum = len(linePoints)
            
            pointsArray = np.array(linePoints)  # x, y
            x = pointsArray[:, 0]
            y = pointsArray[:, 1]
            
            ext_x, ext_y = all_ext_points[c_index]
            
            laneType = self._laneList[c_index]
            # draw prelong lines
            cv2.line(decodeImage, (x[-1], y[-1]), (ext_x, ext_y),  (255, 245, 152), 3)
            for p_index in range(0, pointsNum): 
                cv2.circle(decodeImage, ((x[p_index]), y[p_index]), 3, color_map[laneType], 3)
                
            ## draw ext_points
            cv2.circle(decodeImage, (ext_x, ext_y), 2, (255, 245, 152), 2)
            
            laneMap = ["l1", "r1", "mid"]
            for i in range(0, len(targetPoints)):
                if len(targetPoints[i]) > 0:
                    x, y = targetPoints[i]
                    cv2.circle(decodeImage, (int(x), int(y)), 10, color_map[laneMap[i]], 10)

        cv2.imwrite(self._path + self._imgName + "_ext.jpg", decodeImage)       

    def processLane(self, predImg, srcImg, imgName, drawFlag):
        if len(predImg.shape) == 3:
            predImg = predImg[:, :, 0]
        if len(predImg.shape) == 1:
            predImg = predImg

        self._imgName = imgName
        self.getRegionLabel(predImg)
        #print("regionLabel: ", np.unique(self._regionLabels))
        
        self.getLanePoints()

        height, width, _= srcImg.shape
        self.getExtPoints(height)
        #print("all_ext_points: ", self._all_ext_points, self._laneList)
        
        if len(self._laneList) > 0:
            targetLanes = self.getTargetLaneExtPoints()
        else:
            targetLanes = None
        #print("targetLane: ", targetLanes)

        if drawFlag:
            self.drawPredPoints(srcImg)
            self.drawExtPoints(srcImg, self._all_ext_points, targetLanes)
            #self.drawTargetlaneExtPoints(srcImg, targetLanes)
            #cv2.imwrite(self._path + self._imgName + "_region.jpg", self._regionLabels*32)
        
        return targetLanes

    def processLaneBatch(self, predPath, srcPath, drawFlag):
        srcimglist = self.parseImglist(srcPath)
        predimglist = self.parseImglist(predPath)
        imgNum = len(srcimglist)
        for img_index in range(0, imgNum):
            line_src = srcimglist[img_index].strip('\n')
            line_pred = predimglist[img_index].strip('\n')
            
            srcImage = self.readImage(line_src)
            predImage = self.readImage(line_pred)

            #print("imgname: ", line_pred)
            tmp = line_src.strip('\n').split('/')
            
            imgName = tmp[-1]
            self.processLane(predImage, srcImage, imgName, drawFlag)
        
if __name__ == "__main__":
    savepath = "/Users/xiangsun/Documents/regresLine/testImg/drawRes/badcase/"
    lane_pro = laneProcess(savepath, ratio=8)
   
    """
    predPath = "/Users/xiangsun/Documents/regresLine/testImg/bad1.png"
    srcPath = "/Users/xiangsun/Documents/regresLine/testImg/bad1_src.jpg"
    predImg = lane_pro.readImage(predPath)
    srcImg = lane_pro.readImage(predPath)
    #lane_pro.processLane(predImg, srcImg, "bad1", True)
    """

    predPath = "/Users/xiangsun/Documents/regresLine/testImg/badcase_pred.list"
    srcPath = "/Users/xiangsun/Documents/regresLine/testImg/badcase_src.list"
    
    lane_pro.processLaneBatch(predPath, srcPath, True)

