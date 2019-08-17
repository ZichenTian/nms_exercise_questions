#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "rect.hpp"

extern std::vector<rect> cut_low_score(std::vector<rect> det_results);
extern std::vector<rect> nms(std::vector<rect> det_results);

rect parse_result(std::string line) {
    rect r;
    sscanf(line.data(), "%d%d%d%d%f", &r.x1, &r.y1, &r.x2, &r.y2, &r.score);
    return r;
}

void draw_rect(cv::Mat& image, rect r) {
    // std::cout << r.x1 << " " << r.y2 << " " << r.x2 << " " << r.y2 << " " << std::endl;
    if(r.x1 < 0 || r.x2 < 0 || r.y1 < 0 || r.y2 < 0) {
        std::cerr << "out of range" << std::endl;
        return;
    }
    if(r.x1 >= image.cols || r.x2 >= image.cols || 
       r.y1 >= image.rows || r.y2 >= image.rows) {
        std::cerr << "out of range" << std::endl;
        return;
    }
    if(r.x1 >= r.x2 || r.y1 >= r.y2 || r.score < 0 || r.score > 1) {
        std::cerr << "invalid rect" << std::endl;
        return;
    }
    cv::Rect2i cvr(r.x1, r.y1, r.x2-r.x1, r.y2-r.y1);
    cv::rectangle(image, cvr, cv::Scalar(0,0,255));
    cv::putText(image, std::to_string(r.score), cv::Point(r.x1,r.y1), 
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
    return;
}

void draw_and_show(cv::Mat& srcImage, std::vector<rect>& rects, 
                   std::string window_name, int wait = 0) {
    cv::Mat showImage = srcImage.clone();
    for(auto r : rects) {
        draw_rect(showImage, r);
    }
    cv::imshow(window_name, showImage);
    cv::waitKey(wait);
}

int main(void) {
    std::ifstream result_file("../det_result.txt");
    std::string line;
    std::vector<rect> result_rects;
    while(getline(result_file, line)) {
        result_rects.push_back(parse_result(line));
    }
    std::ifstream label_file("../label.txt");
    std::vector<rect> label_rects;
    while(getline(label_file, line)) {
        label_rects.push_back(parse_result(line));
    }

    cv::Mat srcImage = cv::imread("../timg.jpeg");
    draw_and_show(srcImage, result_rects, "oriImage");
    draw_and_show(srcImage, label_rects, "labelImage");

    std::vector<rect> cut_result_rects = cut_low_score(result_rects);
    draw_and_show(srcImage, cut_result_rects, "cutImage");

    std::vector<rect> final_results = nms(cut_result_rects);
    draw_and_show(srcImage, final_results, "finalImage");

    return 0;
}