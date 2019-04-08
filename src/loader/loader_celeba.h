//
// Created by chenyuan on 18-1-24.
//

#ifndef LOADER_CELEBA_H
#define LOADER_CELEBA_H


#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "helper/bounding_box.h"
#include "video.h"
#include "video_loader.h"

// An image annotation.
struct CelebAAnnotation {
    // Relative path of the image files (must be appended to path).
    std::string image_path;

    // Bounding box annotation for an object in this image.
    BoundingBox bbox;
};


class LoaderCelebA {
public:
    // Load all annotations.
    LoaderCelebA(const std::string& image_folder,
                      const std::string& annotations_folder);

    // Load the specified image.
    void LoadImage(const size_t image_num, cv::Mat* image) const;

    // Get the annotation and the image.
    void LoadAnnotation(const size_t image_num,
                        cv::Mat* image,
                        BoundingBox* bbox) const;
    const std::vector<CelebAAnnotation>& get_images()
    {
        return annotations_;
    }
private:
    std::vector<CelebAAnnotation> annotations_;

};


#endif //GOTURN_LOADER_CELEBA_H
