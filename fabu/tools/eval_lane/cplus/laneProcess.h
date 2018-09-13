#ifndef LANEPROCESS_H
#define LANEPROCESS_H

#include <opencv2/opencv.hpp>

#define DEBUG_ON

class laneProcess {
public:
    laneProcess();
    ~laneProcess();
   
    std::vector<cv::Point> process(const cv::Mat& predImg, const cv::Mat& srcImg);
    void setSaveParam(std::string name, std::string savepath);

private:
    void getSigleLaneRegion(const cv::Mat& predImg, int laneValue, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i>& hierarchy);
    void getAllLaneRegions(const cv::Mat& predImg);    

    std::vector<cv::Point> getTargetExtPoints(std::vector<cv::Point>);
    std::vector<cv::Point> getExtPoints(const cv::Mat& srcImg);
    cv::Point filterExtPoints(std::vector<cv::Point> targetPoints, std::string filter);
    cv::Point getMaxYPoint(std::vector<cv::Point> sigleContourPoints);

    std::string _imageName;
    std::string _savepath;

    int _areaThreshold = 4;
    int _scaleRatio = 8;

    int _laneList[5] = {1, 2, 3, 4, 5};
    int _laneTypeNum = 5;

    std::vector<std::vector<cv::Point> > _allContours;
    std::vector<cv::Vec4i> _allHierarchy;
    std::vector<int> _allLaneList;

    std::map<int, cv::Scalar> _laneColorsMap = {
      {1, cv::Scalar(0, 0, 255)},   // l1 -- red
      {2, cv::Scalar(255, 0, 0)},   // r1 -- blue
      {3, cv::Scalar(0, 255, 0)},   // l2 -- green
      {4, cv::Scalar(0, 255, 255)},  // r2 -- yellow
      {5, cv::Scalar(255, 48, 155)}  // mid -- purple
    };

}; //
#endif
