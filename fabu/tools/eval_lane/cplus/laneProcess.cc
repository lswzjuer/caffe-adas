#include <iostream>

#include "laneProcess.h"

laneProcess::laneProcess() {}
laneProcess::~laneProcess() {}


/// \brief Extract extension point by processing predicted image.
/// \param predImg Predicted grayscale image.
/// \param srcImg Source RGB image.
/// \return targetExtPoints Extension points of lane L1, R1 and Mid.
std::vector<cv::Point> laneProcess::process(const cv::Mat& predImg, const cv::Mat& srcImg) {
    getAllLaneRegions(predImg);
    std::vector<cv::Point> allExtPoints;
    std::vector<cv::Point> targetExtPoints;
    allExtPoints = getExtPoints(srcImg);
    targetExtPoints = getTargetExtPoints(allExtPoints);

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

/// \brief Get connected region of one kind of lane.
/// \param predImg
/// \param laneValue The specified lane label
/// \param contours The contours which keeps all point sets.
/// \param hierarchy The hierarchy of contours.
void laneProcess::getSigleLaneRegion(const cv::Mat& predImg, int laneValue, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i>& hierarchy) {
    //std::cout << "sigle lane: " << laneValue << std::endl;
    cv::Mat img;
    img = predImg.clone();

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

/// \brief Get all connected regions of all lanes.
void laneProcess::getAllLaneRegions(const cv::Mat& predImg) {
    //std::cout << "get all lane regions." << std::endl;
    _allContours.clear();
    _allHierarchy.clear();
    _allLaneList.clear();

    for(int i = 0; i < _laneTypeNum; i++) {
        int laneValue = _laneList[i];
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

/// \brief Get all extension points of the detected lane.
std::vector<cv::Point> laneProcess::getExtPoints(const cv::Mat& srcImg){
#ifdef DEBUG_ON 
    cv::Mat dst = srcImg.clone(); 
#endif
    int height = srcImg.rows;
    int laneNum = _allContours.size();
    std::vector<cv::Point> allExtPoints;
    for(int i = 0; i < laneNum; i++) {
        double slope = 0;
        cv::Point point, extPoint;
        cv::Vec4f lines;

        cv::fitLine(_allContours[i], lines, CV_DIST_L2, 0, 1e-2, 1e-2);

        point.x = lines[2] * _scaleRatio;
        point.y = lines[3] * _scaleRatio;
        slope = lines[1] / lines[0];
        
        extPoint.y = height;
        extPoint.x = point.x + (extPoint.y - point.y) / slope;
        //std::cout << "Points: " << point << " " << extPoint << std::endl;

        allExtPoints.push_back(extPoint);

#ifdef DEBUG_ON
        int pointsNum = _allContours[i].size();
        int laneValue = _allLaneList[i];
        cv::Scalar color = _laneColorsMap[laneValue];
        //std::cout << "laneValue -- colors: " << laneValue << " -- " << color << std::endl;  
        std::vector<cv::Point> rescaleContour;
        std::vector<std::vector<cv::Point> > tmpContour;
        for(int j = 0; j < pointsNum; j++) {
            cv::Point tmpPoint;
            tmpPoint.x = _allContours[i][j].x * _scaleRatio;
            tmpPoint.y = _allContours[i][j].y * _scaleRatio;
            rescaleContour.push_back(tmpPoint);
            //cv::circle(dst, tmpPoint, 3, color, 3);
        }
        tmpContour.push_back(rescaleContour);
        cv::drawContours(dst, tmpContour, 0, color, CV_FILLED);
        
        cv::Point line_start;
        line_start = getMaxYPoint(rescaleContour);
        cv::line(dst, line_start, extPoint, cv::Scalar(255, 255, 0), 5, CV_AA); // cyan -- ext
        cv::circle(dst, extPoint, 15, color, 8); // cyan -- ext
#endif
    }

#ifdef DEBUG_ON
    //cv::imwrite(_savepath + "/" + _imageName + "_contours.jpg", dst); 
    
    std::vector<cv::Point> targetPoints;
    targetPoints = getTargetExtPoints(allExtPoints);
    for(int i = 0 ; i < targetPoints.size(); i++){
        cv::Point extPoint = targetPoints[i];
        if (extPoint.y != -1){
            cv::circle(dst, extPoint, 15, cv::Scalar(255, 255, 0), 8);   
        }
    }
    cv::imwrite(_savepath + "/" + _imageName + "_contours_1.jpg", dst);    
#endif   
    return allExtPoints;
}

/// \brief Filter all extension points to get the needed points of lane l1/r1/mid.
/// \param allExtPoints All extension points.
/// \return The need points.
std::vector<cv::Point> laneProcess::getTargetExtPoints(std::vector<cv::Point> allExtPoints){
    int finalLaneNum = allExtPoints.size();
    std::vector<cv::Point> L1, R1, Mid;
    for(int i = 0; i < finalLaneNum; i++){
        int laneValue = _allLaneList[i];
        if (laneValue == 1) {
            L1.push_back(allExtPoints[i]);
        }
        if (laneValue == 2) {
            R1.push_back(allExtPoints[i]);
        }
        if (laneValue == 5) {
            Mid.push_back(allExtPoints[i]);
        }
    }

    cv::Point l1, r1, mid;
    if(L1.size() > 0){
        l1 = filterExtPoints(L1, "larger");
    }
    else{
        l1.x = l1.y = -1;
    }
        
    if(R1.size() > 0){
        r1 = filterExtPoints(R1, "smaller");
    }
    else{
        r1.x = r1.y = -1;
    }

    if(Mid.size() > 0) {
        mid = filterExtPoints(Mid, "smaller");
    }
    else{
        mid.x = mid.y = -1;
    }

    std::cout << "l1: " << l1 << "  r1: " << r1 << "  mid: " << mid << std::endl;
    std::vector<cv::Point> finalExtPoints;
    finalExtPoints.push_back(l1);
    finalExtPoints.push_back(r1);
    finalExtPoints.push_back(mid);

    return finalExtPoints;
}

/// \brief Choose one point as final one of the same lane by different filter method
/// \param targetPoints The extension points of the same kind lane.
/// \param filter The filter method. "larger" or "smaller".
cv::Point laneProcess::filterExtPoints(std::vector<cv::Point> targetPoints, std::string filter) {
    cv::Point finalPoint, maxPoint, minPoint;

    //std::cout << "target points size: " << targetPoints.size() << std::endl;
    maxPoint = targetPoints[0];
    minPoint = targetPoints[0];
    for(int i = 1; i < targetPoints.size(); i++){
        if(targetPoints[i].x > maxPoint.x) {
            maxPoint.x = targetPoints[i].x;
            maxPoint.y = targetPoints[i].y;
        }
        if(targetPoints[i].x < minPoint.x) {
            minPoint.x = targetPoints[i].x;
            minPoint.y = targetPoints[i].y;
        }
    }
    
    cv::Point tmpMax = maxPoint;
    cv::Point tmpMin = minPoint;
    for(int i = 0; i < targetPoints.size(); i++) {
        //std::cout << "diff: " << tmpMax.x << " " << targetPoints[i].x << std::endl;
        if(abs(tmpMax.x - targetPoints[i].x) < _xDiffThreshold){
            maxPoint.x = (maxPoint.x + targetPoints[i].x) / 2;
        }
        
        if(abs(targetPoints[i].x - tmpMin.x) < _xDiffThreshold){
            minPoint.x = (minPoint.x + targetPoints[i].x) / 2;
        }
    }
    if(filter == "larger"){
        finalPoint = maxPoint;
    }
    if(filter == "smaller"){
        finalPoint = minPoint;
    }
    return finalPoint;
}

/// \brief Get the point whose y value is largest in one connected region. This will only be used in DEBUG_ON
cv::Point laneProcess::getMaxYPoint(std::vector<cv::Point> sigleContourPoints) {
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
