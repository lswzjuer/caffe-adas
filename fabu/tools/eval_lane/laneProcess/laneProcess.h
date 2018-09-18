#ifndef LANEPROCESS_H
#define LANEPROCESS_H

#include <opencv2/opencv.hpp>

#define DEBUG_ON

class laneProcess {
public:
    laneProcess();
    ~laneProcess();
    
    // Process the predicted result and get the extension points of each lane.
    std::vector<cv::Point> process(const cv::Mat& predImg, const cv::Mat& srcImg);
    void setSaveParam(std::string name, std::string savepath);

private:
    // Find all connected region of all lanes. The corresponding result will be saved in _allContours, _allHierarchy and _allLaneList.
    void getAllLaneRegions(const cv::Mat& predImg); 
    // For each lane, find the connected region
    void getSigleLaneRegion(const cv::Mat& predImg, int laneValue, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i>& hierarchy);
   
    // Get the target extension points and corresponding line functions.(L1, R1, Mid)
    // The result will be saved in finalExtPoints and fitlines.
    void getTargetExtPoints(const std::vector<cv::Point> allExtPoints, std::vector<cv::Point>& finalExtPoints, std::vector<cv::Vec4f>& fitlines);
    // Return all extension points of all detected lanes.  
    std::vector<cv::Point> getExtPoints(const cv::Mat& srcImg);
    // Get the final extension points and line functions of the merged lanes. The result will be saved in bestPoint and bestLine.
    void getMergedExtPoint(const std::vector<cv::Point> extPoints, const std::vector<std::vector<cv::Point> > extContours, const std::vector<cv::Vec4f> extLines, const std::string filterType, cv::Point& bestPoint, cv::Vec4f& bestLine);
    // Judge whether some lane needed to be merged and return the merged lanes index.
    std::vector<std::vector<int> > getMergePair(const std::vector<cv::Point> extPoints, const std::vector<std::vector<cv::Point> > contours);
    // For those merged lanes, combinate all points of contours into a new contours and re-fit a line. 
    void reFitLine(const std::vector<std::vector<cv::Point> > contours,  const std::vector<std::vector<int> > allMergePair, const std::string filterType, cv::Point& targetExtPoint, cv::Vec4f& targetLine);   
    // For those unmerged extension points, choose one by filterType  
    void filterExtPoints(const std::vector<cv::Point> targetExtPoints, const std::vector<cv::Vec4f> targetFitlines, const std::string filterType, cv::Point& finalPoint, cv::Vec4f& finalLine);     // The function will be used only in DEBUG_ON for draw prelong line. 
    cv::Point getMaxYPoint(const std::vector<cv::Point> sigleContourPoints);

    std::string _imageName;
    std::string _savepath;

    int _areaThreshold = 6;
    int _xDiffThreshold = 40;
    int _scaleRatio = 8;

    int _initLaneList[5] = {1, 2, 3, 4, 5};
    int _initLaneTypeNum = 5;

    // for extension points
    int _height;

    // points of connected region 
    std::vector<std::vector<cv::Point> > _allContours;
    // hierarchy of each connected region, used for drawContours
    std::vector<cv::Vec4i> _allHierarchy;
    // value list of choosed lane
    std::vector<int> _allLaneList;

    // store the slopes of fitted line
    std::vector<cv::Vec4f> _allFitLines;

    // color map for draw different lane
    std::map<int, cv::Scalar> _laneColorsMap = {
      {1, cv::Scalar(0, 0, 255)},   // l1 -- red
      {2, cv::Scalar(255, 0, 0)},   // r1 -- blue
      {3, cv::Scalar(0, 255, 0)},   // l2 -- green
      {4, cv::Scalar(0, 255, 255)},  // r2 -- yellow
      {5, cv::Scalar(255, 48, 155)}  // mid -- purple
      // cyan -- extension lane
    };

}; //
#endif
