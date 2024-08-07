/**
 * @file VideoProcessor.cpp
 * @author Laura Scarabello (laura.scarabello.1@studenti.unipd.it)
 * @brief # 2024175
 * @date 2024-06-25
 * 
 */

#include "VideoProcessor.h"
#include <iostream>

VideoProcessor::VideoProcessor(const std::string& videoPath) : opened(false) {
    cap.open(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
    } else {
        opened = true;
    }
}

VideoProcessor::~VideoProcessor() {
    if (cap.isOpened()) {
        cap.release();
    }
}

bool VideoProcessor::isOpened() const {
    return opened;
}

cv::Mat VideoProcessor::getNextFrame() {
    cv::Mat frame;
    if (opened) {
        if (!cap.read(frame)) {
            std::cerr << "No more frames to read." << std::endl;
            opened = false;
        }
    }
    return frame;
}
