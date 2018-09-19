#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <net.h>
#include "cpu.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "exec img_list outpath" << endl;
        return 0;
    }
    string inTxt = argv[1];
    string outpath = argv[2];
    ncnn::set_omp_num_threads(0);
    std::string filename(inTxt.c_str());
    std::ifstream file;
    file.open(filename, std::ios::in);
    
    std::string imgfile;
    ncnn::Net net;
    net.load_param("shufflenet.param");
    net.load_model("shufflenet.bin");
    std::cout<<"net loaded"<<std::endl;
    int count = 1;
    float img_scale[3] = {0.017, 0.017, 0.017}, img_bias[3] = {103.94, 116.78, 123.68}; 
    // ncnn::set_omp_dynamic(0);
    while(getline(file, imgfile)){
        cv::Mat cv_img=cv::imread(imgfile, CV_LOAD_IMAGE_COLOR);
        cv::Mat resized_img;
        cv::resize(cv_img, resized_img,cv::Size(50, 50));
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(resized_img.data, ncnn::Mat::PIXEL_BGR, resized_img.cols, resized_img.rows);
        ncnn_img.substract_mean_normalize(img_bias, img_scale);
        //matout(ncnn_img);
        ncnn::Extractor ex = net.create_extractor();
        ex.set_num_threads(8);
        ex.input("data", ncnn_img);
        ncnn::Mat out;
        ex.extract("prob", out);
        // Detect human face bounding boxes of input image.
        count++;
        int w = out.w;
        int max_index = 0;
        float max_score = 0;
        string class_type[] = {"Normal: ", "Phone: ", "Drink: ", "Smoke: "};
        for (int row = 0; row < w; ++row) {
            if (out[row] > max_score) {
                max_score = out[row];
                max_index = row;
            }
        }
        for (int row = 0; row < w; ++row) {
            if (out[row] == max_score) {
                putText(cv_img, class_type[row] + to_string(out[row]), Point(50, 75 * (row + 1)), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 2);
            }
            else {
                putText(cv_img, class_type[row] + to_string(out[row]), Point(50, 75 * (row + 1)), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
            }
        } 
        int label = 0;
        if (imgfile.find("phone") != string::npos) {
            label = 1;
        }
        else if (imgfile.find("drink") != string::npos) {
            label = 2;
        }
        else if (imgfile.find("smoke") != string::npos) {
            label = 3;
        }

        string img_name = outpath + imgfile.substr(imgfile.find_last_of("/"), imgfile.size());
        if (label != max_index) {
            imwrite(img_name, cv_img);
        }
        std::cout << "For " << count << "th img, the index and score is " <<max_index << " " << max_score << std::endl;
    }
    
    return 0;
}
