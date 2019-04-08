//
// Created by chenyuan on 18-1-24.
//

#ifndef LOADER_MSCELEB_H
#define LOADER_MSCELEB_H


#include <vector>
#include <string>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "helper/bounding_box.h"
#include "video.h"
#include "video_loader.h"

// An image annotation.
struct MSCelebAnnotation {
    // Relative path of the image files (must be appended to path).
    std::string image_path;

    // Bounding box annotation for an object in this image.
    BoundingBox bbox;
};


class LoaderMSCeleb {
public:
    // Load all annotations.
    LoaderMSCeleb(const std::string& image_folder,
                      const std::string& annotations_folder);

    // Load the specified image.
    void LoadImage(const size_t image_num, cv::Mat* image) const;

    // Get the annotation and the image.
    void LoadAnnotation(const size_t image_num,
                        cv::Mat* image,
                        BoundingBox* bbox) const;
    const std::vector<MSCelebAnnotation>& get_images()
    {
        return annotations_;
    }

    bool isTheSameDir(int num1, int num2) const
    {
        if(num1>0&&num1<annotations_.size()&&num2>0&&num2<annotations_.size())
        {
            MSCelebAnnotation a1=annotations_[num1];
            MSCelebAnnotation a2=annotations_[num2];

            std::size_t  pos1 = a1.image_path.find_last_of('/');
            //std::cout << " path: " << a1.image_path.substr(0,pos1) << '\n';

            std::size_t  pos2 = a2.image_path.find_last_of('/');
            //std::cout << " path: " << a2.image_path.substr(0,pos2) << '\n';

            if(a1.image_path.substr(0,pos1) == a2.image_path.substr(0,pos2))
            {
                return true;
            }
        }
        return false;
    }
private:
    std::vector<MSCelebAnnotation> annotations_;

};


#endif //GOTURN_LOADER_CELEBA_H
