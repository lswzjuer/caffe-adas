#include <iostream>
#include <sstream>
#include "laneProcess.h"

laneProcess::laneProcess() {}
laneProcess::~laneProcess() {}


/// \brief Extract extension point by processing predicted image.
/// \param [in]predImg Predicted grayscale image.
/// \param [in]srcImg Source RGB image.
/// \return [out]targetExtPoints Extension points of lane L1, R1 and Mid.
std::vector<cv::Point> laneProcess::process(const cv::Mat& predImg, const cv::Mat& srcImg) {
    _height = srcImg.rows;
    getAllLaneRegions(predImg);
    std::vector<cv::Point> allExtPoints;
    std::vector<cv::Point> targetExtPoints;
    // point-slope form
    // lines[0]-->dx  lines[1]-->dy lines[2]-->point.x lines[3]-->point.y
    std::vector<cv::Vec4f> targetLines;
    allExtPoints = getExtPoints(srcImg);
    getTargetExtPoints(allExtPoints, targetExtPoints, targetLines);

#ifdef DEBUG_ON
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(predImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    
    cv::Mat dst = cv::Mat::zeros(predImg.rows, predImg.cols, CV_8UC3);
    int idx = 0;
    std::cout << "hierarchy size: " << contours.size() << " " << hierarchy.size() << std::endl;
    for(int i = 0; i < contours.size(); i++) {
        for(; idx >=0; idx = hierarchy[idx][0]){
            cv::Scalar color(rand()&255, rand()&255, rand()&255);
            cv::drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
        }
    }
    cv::imwrite(_savepath + "/" + _imageName + "_raw.jpg", dst);
#endif

    return targetExtPoints;
}

/// \brief Get all connected regions of all lanes.
/// \param [in]predImg The predicted grayscale image.
/// \param [out]_allContours The points of all lanes
/// \param [out]_allHierarchy The hierarchy architecture of connected region. This variable is used to in drawContours when DEBUG_ON is open.
/// \param [out]_allLaneList The kept lane values after filtering the connected region area. 1-->L1, 2-->R1, 3-->L2, 4-->R2, 5-->Mid
void laneProcess::getAllLaneRegions(const cv::Mat& predImg) {
    //std::cout << "get all lane regions." << std::endl;
    _allContours.clear();
    _allHierarchy.clear();
    _allLaneList.clear();

    for(int i = 0; i < _initLaneTypeNum; i++) {
        int laneValue = _initLaneList[i];
        //std::cout << "lane value: " << laneValue << std::endl;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;   
        getSigleLaneRegion(predImg, laneValue, contours, hierarchy);
       
        for(int j = 0; j < contours.size(); j ++) {
            //std::cout << "sigle contours: " << cv::contourArea(contours[j]) << " " << contours[j].size() << " " << _areaThreshold << std::endl;
            if(cv::contourArea(contours[j]) >= _areaThreshold) {
                _allContours.push_back(contours[j]);
                _allHierarchy.push_back(hierarchy[j]);
                _allLaneList.push_back(laneValue);
            }
        }
    }
    //std::cout << "coutours final size: " << _allContours.size()  << std::endl;
 }

/// \brief Get connected region of one kind of lane. This function will be called in getAllLaneRegions.
/// \param [int]predImg
/// \param [in]laneValue The specified lane label
/// \param [out]contours The contours which keeps all point sets.
/// \param [out]hierarchy The hierarchy of contours.
void laneProcess::getSigleLaneRegion(const cv::Mat& predImg, int laneValue, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i>& hierarchy) {
    //std::cout << "sigle lane: " << laneValue << std::endl;
    cv::Mat img;
    img = predImg.clone();

    // Change some pixel values for finding best contours.
    for(cv::MatIterator_<uchar> it = img.begin<uchar>(); it != img.end<uchar>(); it++) {
        int pixel = *it;
        //pixel = pixel / 32;
        if(pixel != laneValue){
            *it = 0;
        }
    }
    //std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
}

/// \brief Get all extension points of the detected lane.
/// \param [in]srcImg 
/// \param [out]_allFitLines The fitted line function for all kept lanes.
/// \return allExtPoints The extension points of all kept lanes.
std::vector<cv::Point> laneProcess::getExtPoints(const cv::Mat& srcImg){
#ifdef DEBUG_ON 
    cv::Mat dst = srcImg.clone(); 
#endif
    _allFitLines.clear();
    int height = srcImg.rows;
    int laneNum = _allContours.size();
    std::vector<cv::Point> allExtPoints;
    // Get one extension point and one line function for each kind of lane.
    for(int i = 0; i < laneNum; i++) {
        double slope = 0;
        cv::Point point, extPoint;
        cv::Vec4f lines;

        // point-slope form
        // lines[0]-->dx  lines[1]-->dy lines[2]-->point.x lines[3]-->point.y
        cv::fitLine(_allContours[i], lines, CV_DIST_L2, 0, 1e-2, 1e-2);

        point.x = lines[2] * _scaleRatio;
        point.y = lines[3] * _scaleRatio;
        slope = lines[1] / lines[0];
        
        extPoint.y = height;
        extPoint.x = point.x + (extPoint.y - point.y) / slope;
        //std::cout << "Points: " << point << " " << extPoint << std::endl;

        allExtPoints.push_back(extPoint);
        _allFitLines.push_back(lines);

#ifdef DEBUG_ON
        // draw the predicted point and lines
        int pointsNum = _allContours[i].size();
        int laneValue = _allLaneList[i];
        cv::Scalar color = _laneColorsMap[laneValue];
        //std::cout << "laneValue -- colors: " << laneValue << " -- " << color << std::endl;  
        std::vector<cv::Point> rescaleContour;
        std::vector<std::vector<cv::Point> > tmpContour;
        for(int j = 0; j < pointsNum; j++) {
            cv::Point tmpPoint;
            // Map the coordinate to source image.
            tmpPoint.x = _allContours[i][j].x * _scaleRatio;
            tmpPoint.y = _allContours[i][j].y * _scaleRatio;
            rescaleContour.push_back(tmpPoint);
            //cv::circle(dst, tmpPoint, 3, color, 3);
        }
        tmpContour.push_back(rescaleContour);
        cv::drawContours(dst, tmpContour, 0, color, CV_FILLED);
        
        cv::Point line_start;
        // Choose the maximum of y value as the start point of line.
        line_start = getMaxYPoint(rescaleContour);
        cv::line(dst, line_start, extPoint, cv::Scalar(255, 255, 0), 5, CV_AA); // cyan -- ext
        cv::circle(dst, extPoint, 15, color, 8); // cyan -- ext
#endif
    }

#ifdef DEBUG_ON
    std::cout << "First ext points: " << allExtPoints << std::endl;
    //cv::imwrite(_savepath + "/" + _imageName + "_contours.jpg", dst); 
    
    // draw extension points
    std::vector<cv::Point> targetPoints;
    std::vector<cv::Vec4f> fitlines;
    getTargetExtPoints(allExtPoints, targetPoints, fitlines);
    int targetColor[3] = {1, 2, 5};
    std::string targetName[3]  ={"l1", "r1", "mid"};
    for(int i = 0 ; i < targetPoints.size(); i++){
        cv::Scalar color = _laneColorsMap[targetColor[i]];
        std::string lane = targetName[i];
        cv::Point extPoint = targetPoints[i];
        if (extPoint.y != -1){
            cv::circle(dst, extPoint, 15, cv::Scalar(255, 255, 0), 8);
            std::ostringstream str;
            str << lane << ": " << extPoint.x << " " << extPoint.y;
            cv::Point left;
            left.x = 30;
            left.y = 30 + i * 30;
            cv::putText(dst, str.str(), left, CV_FONT_HERSHEY_SIMPLEX, 1, color, 2);
        }
    }
    cv::imwrite(_savepath + "/" + _imageName + "_contours_1.jpg", dst);    
    std::cout << "finish get allExtPoints\n";
#endif
    return allExtPoints;
}

/// \brief Filter all extension points to get the needed points of lane l1/r1/mid.
/// \param allExtPoints All extension points.
/// \return The needed points.
void laneProcess::getTargetExtPoints(const std::vector<cv::Point> allExtPoints, std::vector<cv::Point>& finalExtPoints, std::vector<cv::Vec4f>& fitlines){
    int finalLaneNum = allExtPoints.size();
    
    // Get extension points, contours, fitted line functions of L1/R1/Mid to choose the final result by merging some points. 
    std::vector<cv::Point> L1_extPoints, R1_extPoints, Mid_extPoints;
    std::vector<std::vector<cv::Point> > L1_contours, R1_contours, Mid_contours;
    std::vector<cv::Vec4f> L1_fitlines, R1_fitlines, Mid_fitlines;
    for(int i = 0; i < finalLaneNum; i++){
        int laneValue = _allLaneList[i];
        if (laneValue == 1) {
            L1_extPoints.push_back(allExtPoints[i]);
            L1_contours.push_back(_allContours[i]);
            L1_fitlines.push_back(_allFitLines[i]);
        }
        if (laneValue == 2) {
            R1_extPoints.push_back(allExtPoints[i]);
            R1_contours.push_back(_allContours[i]);
            R1_fitlines.push_back(_allFitLines[i]);
        }
        if (laneValue == 5) {
            Mid_extPoints.push_back(allExtPoints[i]);
            Mid_contours.push_back(_allContours[i]);
            Mid_fitlines.push_back(_allFitLines[i]);
        }
#ifdef DEBUG_ON
        std::cout << "first line func: " << _allFitLines[i] << std::endl; 
#endif
    }

    std::vector<std::vector<int> > mergePair;
    cv::Point l1_extPoint, r1_extPoint, mid_extPoint;
    cv::Vec4f l1_line, r1_line, mid_line;

    //std::cout << "l1 size: " << L1_extPoints.size() << " " << L1_contours.size() << " " << L1_fitlines.size() << std::endl;

    if(L1_extPoints.size() > 0) {
       // More than zero L1 extension points
       getMergedExtPoint(L1_extPoints, L1_contours, L1_fitlines, "larger", l1_extPoint, l1_line);
    }else{
        l1_extPoint.x = l1_extPoint.y = -1;
        l1_line = {-1, -1, -1, -1};
    }

    if(R1_extPoints.size() > 0) {
       getMergedExtPoint(R1_extPoints, R1_contours, R1_fitlines, "smaller", r1_extPoint, r1_line);
    }else{
        r1_extPoint.x = r1_extPoint.y = -1;
        r1_line = {-1, -1, -1, -1};
    }

    if(Mid_extPoints.size() > 0) {
       getMergedExtPoint(Mid_extPoints, Mid_contours, Mid_fitlines, "smaller", mid_extPoint, mid_line);
    }else{
        mid_extPoint.x = mid_extPoint.y = -1;
        mid_line = {-1, -1, -1, -1};
    }    

#ifdef DEBUG_ON
    std::cout << "l1: " << l1_extPoint << "  r1: " << r1_extPoint << "  mid: " << mid_extPoint << std::endl;
    std::cout << "l1_line: " << l1_line << "  r1_line: " << r1_line << "  mid_line: " << mid_line << std::endl;  
#endif

    // store final extension points of L1/R1/Mid 
    finalExtPoints.push_back(l1_extPoint);
    finalExtPoints.push_back(r1_extPoint);
    finalExtPoints.push_back(mid_extPoint);

    // store final fitted line function of L1/R1/Mid     
    fitlines.push_back(l1_line);
    fitlines.push_back(r1_line);
    fitlines.push_back(mid_line);
}

/// \brief Choose the best extension points and the corresponding line function.
/// \param [in] extPoints The extension points of one kind of lane
/// \param [in] extContours All points predicted as the specific lane by NN.
/// \param [in] extLines All fitted line functions of one kind of lane.
/// \param [in] filterType The method to choose points.
/// \param [out] bestPoint The final used extension points.
/// \param [out] bestLine The final used line functions.
void laneProcess::getMergedExtPoint(const std::vector<cv::Point> extPoints, const std::vector<std::vector<cv::Point> > extContours, const std::vector<cv::Vec4f> extLines, const std::string filterType, cv::Point& bestPoint, cv::Vec4f& bestLine) {
    std::vector<std::vector<int> > mergePair;
    // find the lane pair that need to be re-fitted
    mergePair = getMergePair(extPoints, extContours);
    // At least one pair to merge
    if(mergePair.size() > 0) {
        reFitLine(extContours, mergePair, filterType, bestPoint, bestLine);
    }else {
        // choose the largest x_value.
        filterExtPoints(extPoints, extLines, filterType, bestPoint, bestLine);
    }
}

/// \brief Judge whether some predicted lanes need to be merged by the difference of x value.
/// \param [in] extPoints 
/// \param [in] contours
/// \return allMergePair The index of those lanes needed to be merged. 
std::vector<std::vector<int> > laneProcess::getMergePair(const std::vector<cv::Point> extPoints, const std::vector<std::vector<cv::Point> > contours){
        int extNum = extPoints.size();
        std::vector<std::vector<int> > allMergePair;
        std::vector<int> sigleMergePair;
       
        //std::cout << "getMergePair begin\n";
        // usually extNum is smaller than 3
        for(int i = 0; i < extNum; i++) {
          for(int j = i + 1; j < extNum; j++) {
              sigleMergePair.clear();
              if (abs(extPoints[i].x - extPoints[j].x) < _xDiffThreshold) {
                 sigleMergePair.push_back(i);
                 sigleMergePair.push_back(j);
                 allMergePair.push_back(sigleMergePair);
              }
          }
        }
        //std::cout << "getMergePair end\n";
        return allMergePair;
    }

/// \brief Re-fit a new line function by combinating the lane contours into a new one.
/// \param [in] contours
/// \param [in] allMergePair 
/// \param [in] filterType
/// \param [out] targetExtPoint The new extension points.
/// \param [out] targetLine The new fitted line functions.
void laneProcess::reFitLine(const std::vector<std::vector<cv::Point> > contours,  const std::vector<std::vector<int> > allMergePair, const std::string filterType, cv::Point& targetExtPoint, cv::Vec4f& targetLine) {
    cv::Point finalPoint, maxPoint, minPoint;
    cv::Vec4f maxLine, minLine;
    std::vector<cv::Point> extPoints;
    std::vector<cv::Vec4f> lines;
   
    // mergePairNum should be more than zero
    int mergePairNum = allMergePair.size();

    for(int i = 0; i < mergePairNum; i++){
        int index1 = allMergePair[i][0];
        int index2 = allMergePair[i][1];
        std::vector<cv::Point> newContours;

        // combinate two contours into one
        newContours.insert(newContours.end(), contours[index1].begin(), contours[index1].end());
        newContours.insert(newContours.end(), contours[index2].begin(), contours[index2].end());        
        double slope = 0;
        cv::Point point, extPoint;
        cv::Vec4f line;

        cv::fitLine(newContours, line, CV_DIST_L2, 0, 1e-2, 1e-2);
        point.x = line[2] * _scaleRatio;
        point.y = line[3] * _scaleRatio;
        slope = line[1] / line[0];

        extPoint.y = _height;
        extPoint.x = point.x + (extPoint.y - point.y) / slope;

        extPoints.push_back(extPoint);
        lines.push_back(line);
    }

    // choose maximum point and minimum point
    maxPoint = extPoints[0];
    minPoint = extPoints[0];
    maxLine = lines[0];
    minLine = lines[0];
    for(int i = 0; i < extPoints.size(); i++) {
        if(extPoints[i].x > maxPoint.x){
            maxPoint = extPoints[i];
            maxLine = lines[i];
        }
        if(extPoints[i].x < minPoint.x) {
            minPoint = extPoints[i];
            minLine = lines[i];
        }
    }

    if(filterType == "larger"){
        targetExtPoint = maxPoint;
        targetLine = maxLine;
    }
    if(filterType == "smaller"){
        targetExtPoint = minPoint;
        targetLine = minLine;
    }
#ifdef DEBUG_ON
    std::cout << "reFit line: " << targetLine << std::endl;
#endif
}

/// \brief Choose one point as final one of the same lane by different filter method
/// \param [in] targetPoints The extension points of the same kind lane.
/// \param [in] targetFitLines The fitted line functions.
/// \param [in] filter The filter method. "larger" or "smaller".
/// \param [out] finalPoint
/// \param [out] finalLine
void laneProcess::filterExtPoints(const std::vector<cv::Point> targetExtPoints, const std::vector<cv::Vec4f> targetFitLines, const std::string filter, cv::Point& finalPoint, cv::Vec4f& finalLine) {
    cv::Point maxPoint, minPoint;
    cv::Vec4f maxLine, minLine;

    //std::cout << "target points size: " << targetExtPoints.size() << " " << targetFitLines.size() << std::endl;
    maxPoint = targetExtPoints[0];
    minPoint = targetExtPoints[0];
    maxLine = targetFitLines[0];
    minLine = targetFitLines[0];
    for(int i = 1; i < targetExtPoints.size(); i++){
        if(targetExtPoints[i].x > maxPoint.x) {
            maxPoint.x = targetExtPoints[i].x;
            maxPoint.y = targetExtPoints[i].y;
            maxLine = targetFitLines[i];
        }
        if(targetExtPoints[i].x < minPoint.x) {
            minPoint.x = targetExtPoints[i].x;
            minPoint.y = targetExtPoints[i].y;
            minLine = targetFitLines[i];
        }
    }
    
    if(filter == "larger"){
        finalPoint = maxPoint;
        finalLine = maxLine;
    }
    if(filter == "smaller"){
        finalPoint = minPoint;
        finalLine = minLine;
    }
}

/// \brief Get the point whose y value is largest in one connected region. This will only be used in DEBUG_ON
cv::Point laneProcess::getMaxYPoint(const std::vector<cv::Point> sigleContourPoints) {
    cv::Point maxYPoint;
    int maxIndex = 0;

    maxYPoint = sigleContourPoints[0];
    for(int i = 1; i < sigleContourPoints.size(); i++){
        if(sigleContourPoints[i].y > maxYPoint.y) {
            maxYPoint.x = sigleContourPoints[i].x;
            maxYPoint.y = sigleContourPoints[i].y;
        }
    }

    long xValueSum = 0;
    int xValueNum = 0;
    for(int i = 0; i < sigleContourPoints.size(); i++){
        if(sigleContourPoints[i].y == maxYPoint.y) {
            xValueSum += sigleContourPoints[i].x;
            xValueNum += 1;
        }
    }
    // take x_axis mean of all maxYPoints 
    maxYPoint.x = (int)xValueSum / (xValueNum + 1e-30);

    return maxYPoint;
}

/// \breif Set some parameters for saving image. 
void laneProcess::setSaveParam(std::string name, std::string savepath){
    this->_imageName = name;
    this->_savepath = savepath;
}
