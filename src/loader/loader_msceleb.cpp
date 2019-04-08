//
// Created by chenyuan on 18-1-24.
//
#include <fstream>
#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "loader_msceleb.h"
#include "helper/helper.h"

// If true, only load a small number of images.
const bool kDoTest = false;
using std::string;
using std::vector;

LoaderMSCeleb::LoaderMSCeleb(const std::string &image_folder, const std::string &annotations_floder)
{

    // Open the annotation file.
    const string& bbox_groundtruth_path = annotations_floder + "/bboxes.txt";
    FILE *annotation_file_ptr = fopen(bbox_groundtruth_path.c_str(), "r");
    //m.0107_f/0_FaceId-0.jpg
    char image_name[32];
    double Ax, Ay, Aw, Ah;
    while (true) {
        const int status = fscanf(annotation_file_ptr, "%s %lf %lf %lf %lf\n",
                                  image_name, &Ax, &Ay, &Aw, &Ah);
        if (status == EOF) {
            break;
        }
        //printf("%s %d %d %d %d\n",image_name, Ax,Ay,Aw,Ah);
        MSCelebAnnotation msca;
        msca.image_path = image_folder+"/"+image_name;
        msca.bbox.x1_ = Ax;
        msca.bbox.y1_ = Ay;
        msca.bbox.x2_ = Aw;
        msca.bbox.y2_ = Ah;
        //std::cout<<ca.image_path.c_str();
        //std::cout<<ca.bbox.x1_ <<ca.bbox.y1_ <<ca.bbox.x2_-ca.bbox.x1_ <<ca.bbox.y2_-ca.bbox.y1_ <<"\n";
        annotations_.push_back(msca);
    }
    fclose(annotation_file_ptr);
}

void LoaderMSCeleb::LoadImage(const size_t image_num, cv::Mat *image) const {
    // Load the first annotation for this image.
    const int annotation_num = 0;
    const MSCelebAnnotation& annotation = annotations_[image_num];

    // Load the specified image (using the file-path contained within the annotation).
    const string& image_file = annotation.image_path;
    *image = cv::imread(image_file.c_str());

    // Check that we were able to load the image.
    if (!image->data) {
        printf("Could not open or find image %s\n", image_file.c_str());
        return;
    }
}

void LoaderMSCeleb::LoadAnnotation(const size_t image_num, cv::Mat *image, BoundingBox *bbox) const {

    // Load the specified image's annotations.

    const MSCelebAnnotation& annotation = annotations_[image_num];

    // Load the specified image.
    const string& image_file = annotation.image_path;
    *image = cv::imread(image_file.c_str());

    // Check that we were able to load the image.
    if (!image->data) {
        printf("Could not open or find image %s\n", image_file.c_str());
        return;
    }
    // Scale the bounding box by the ratio of the the image size to the display size.
    *bbox = annotation.bbox;
}

