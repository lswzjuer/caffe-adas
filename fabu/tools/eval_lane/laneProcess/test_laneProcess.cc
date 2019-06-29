#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <ctime>

#include "laneProcess.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: ./laneProcess predlist savepath \n";
        exit(0);
    }

    std::string predfile = argv[1];
    std::ifstream predlist;
    predlist.open(predfile.c_str(), std::ios::in);

    std::string savepath = argv[2];

    if(!predlist.is_open()) {
        std::cout << "predicted list open failed. \n";
    }

    std::string imgfile;
    while(getline(predlist, imgfile)) {
        std::cout << "imgName: " << imgfile << std::endl;
        std::string imgName = imgfile.substr(imgfile.rfind("/") + 1);

        std::string suffix = "_res.png";
        std::string srcImgName = imgName.substr(0, imgName.rfind("_res1.png")) + "_src.jpg";
        std::string srcImgPath = imgfile.substr(0, imgfile.rfind("/")) + "/" + srcImgName;
        std::cout << "srcImgPath: " << srcImgPath << std::endl;
       
        cv::Mat srcImg = cv::imread(srcImgPath);
        cv::Mat predImg = cv::imread(imgfile);
        cv::cvtColor(predImg, predImg, CV_RGB2GRAY);

        if(!predImg.data) {
            std::cout << "srcImg empty()." << std::endl;
            exit(0);    
        }
        if(!srcImg.data){
            std::cout << "srcImg empty()." << std::endl;
            exit(0);
        }

        std::cout << "img: " << srcImg.size() << std::endl;
        //std::cout << "img size: " << predImg.size() << " " << predImg.channels() << std::endl;

        laneProcess lane_process = laneProcess();
        lane_process.setSaveParam(imgName, savepath);
        std::vector<cv::Point> resExtPoints;
        clock_t start = clock();
        resExtPoints = lane_process.process(predImg, srcImg);
        clock_t end = clock();
        std::cout << "It cost " << (float)(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;
        //lane_process.process(img);
        std::cout << std::endl;
    
    }

}
