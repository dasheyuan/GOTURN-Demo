//
// Created by chenyuan on 18-1-24.
//
#include <time.h>
#include "loader/loader_celeba.h"

using std::string;
using namespace cv;

// Compare two images by getting the L2 error (square-root of sum of squared error).
double getSimilarity( const Mat A, const Mat B ) {
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        // Calculate the L2 relative error between images.
        double errorL2 = norm(A, B, NORM_L2);
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double similarity = errorL2 / (double) (A.rows * A.cols);
        return similarity;
    } else {
        //Images have a different size
        return 100000000.0;  // Return a bad value
    }
}
int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " alov_videos_folder alov_annotations_folder"
                  << std::endl;
        return 1;
    }
    srand(time(0));
    int arg_index = 1;
    const string &celeba_folder = argv[arg_index++];
    const string &celeba_annotations_folder = argv[arg_index++];
    LoaderCelebA loaderCelebA(celeba_folder, celeba_annotations_folder);
    for (int i = 0; i < 10; i++) {

        int image_num = rand() % loaderCelebA.get_images().size();
        int image_num2 = rand() % loaderCelebA.get_images().size();
        cv::Mat img,img2,img_r,img2_r;
        BoundingBox bbox,bbox2;
        loaderCelebA.LoadAnnotation(image_num, &img, &bbox);
        loaderCelebA.LoadAnnotation(image_num2, &img2, &bbox2);

        /*cv::Mat temp(227,227,CV_8UC3,cv::Scalar(255,255,255));
        temp.copyTo(img_r);
        cv::Mat temp2(227,227,CV_8UC3,cv::Scalar(0,0,0));
        temp2.copyTo(img2_r);*/

        cv::Mat img_b(img,cv::Rect(bbox.x1_,bbox.y1_,bbox.x2_-bbox.x1_,bbox.y2_-bbox.y1_));
        cv::Mat img2_b(img2,cv::Rect(bbox2.x1_,bbox2.y1_,bbox2.x2_-bbox2.x1_,bbox2.y2_-bbox2.y1_));
        cv::resize(img_b,img_r,cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
        cv::resize(img2_b,img2_r,cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
        cv::rectangle(img,cvPoint(bbox.x1_,bbox.y1_),cvPoint(bbox.x2_,bbox.y2_),cvScalar(255,255,255));
        std::cout<<getSimilarity(img_r,img_r)<<"\n";
        cv::imshow("1", img_r);
        cv::imshow("2", img2_r);
        cv::waitKey(0);
    }
    return 0;
}