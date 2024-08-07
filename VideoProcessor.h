/**
 * @file VideoProcessor.h
 * @author Laura Scarabello (laura.scarabello.1@studenti.unipd.it)
 * @brief # 2024175
 * @date 2024-06-25
 * 
 */

#ifndef VIDEOPROCESSOR_H
#define VIDEOPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>

class VideoProcessor {
public:
    VideoProcessor(const std::string& videoPath);
    ~VideoProcessor();

    bool isOpened() const;
    cv::Mat getNextFrame();

private:
    cv::VideoCapture cap;
    bool opened;
};

#endif // VIDEOPROCESSOR_H
