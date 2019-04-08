// Train the neural network tracker.

#include <string>
#include <iostream>

#include <caffe/caffe.hpp>

#include "example_generator.h"
#include "helper/helper.h"
#include "loader/loader_imagenet_det.h"
#include "loader/loader_alov.h"
#include "network/regressor_train.h"
#include "train/tracker_trainer.h"
#include "tracker/tracker_manager.h"
#include "loader/video.h"
#include "loader/video_loader.h"
#include "loader/loader_celeba.h"

using std::string;

// Desired number of training batches.
const int kNumBatches = 500000;

namespace {

    // Compare two images by getting the L2 error (square-root of sum of squared error).
    double getSimilarity( const cv::Mat A, const cv::Mat B ) {
        if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
            // Calculate the L2 relative error between images.
            double errorL2 = cv::norm(A, B, cv::NORM_L2);
            // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
            double similarity = errorL2 / (double) (A.rows * A.cols);
            return similarity;
        } else {
            //Images have a different size
            return 100000000.0;  // Return a bad value
        }
    }

// Train on a random image.
    void train_image(const LoaderCelebA &image_loader,
                     const std::vector<CelebAAnnotation> &images,
                     TrackerTrainer *tracker_trainer) {
        cv::Mat image,image2;
        BoundingBox bbox,bbox2;
        cv::Mat img_r,img2_r;

        srand((unsigned)time( NULL ));
        int image_num = rand() % (images.size()-2599);
        int image_num2 = rand() % (images.size()-2599);
        image_loader.LoadAnnotation(image_num, &image, &bbox);
        image_loader.LoadAnnotation(image_num2, &image2, &bbox2);

        cv::Mat img_b(image,cv::Rect(bbox.x1_,bbox.y1_,bbox.x2_-bbox.x1_,bbox.y2_-bbox.y1_));
        cv::Mat img2_b(image2,cv::Rect(bbox2.x1_,bbox2.y1_,bbox2.x2_-bbox2.x1_,bbox2.y2_-bbox2.y1_));
        cv::resize(img_b,img_r,cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);
        cv::resize(img2_b,img2_r,cv::Size(227, 227), (0, 0), (0, 0), cv::INTER_LINEAR);


        if(getSimilarity(img_r,img2_r)>0.7)
        {
            /*cv::imshow("1", img_r);
            cv::imshow("2", img2_r);
            cv::waitKey(30);*/
            // Train on this example
            bbox.p_ = 1;
            tracker_trainer->Train(img_r, img_r, bbox, bbox);
            bbox2.p_ = 1;
            tracker_trainer->Train(img2_r, img2_r, bbox2, bbox2);
            bbox2.p_ = 0;
            tracker_trainer->Train(img_r, img2_r, bbox, bbox2);
            bbox.p_= 0;
            tracker_trainer->Train(img2_r, img_r, bbox2, bbox);
        }


    }

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " videos_folder_imagenet annotations_folder_imagenet"
                  << " alov_videos_folder alov_annotations_folder"
                  << " network.caffemodel train.prototxt val.prototxt"
                  << " solver_file"
                  << " lambda_shift lambda_scale min_scale max_scale"
                  << " gpu_id"
                  << std::endl;
        return 1;
    }

    FLAGS_alsologtostderr = 1;

    ::google::InitGoogleLogging(argv[0]);

    int arg_index = 1;
    const string &videos_folder_celeba = argv[arg_index++];
    const string &annotations_folder_celeba = argv[arg_index++];
    const string &caffe_model = argv[arg_index++];
    const string &train_proto = argv[arg_index++];
    const string &solver_file = argv[arg_index++];
    const double lambda_shift = atof(argv[arg_index++]);
    const double lambda_scale = atof(argv[arg_index++]);
    const double min_scale = atof(argv[arg_index++]);
    const double max_scale = atof(argv[arg_index++]);
    const int gpu_id = atoi(argv[arg_index++]);
    const int random_seed = atoi(argv[arg_index++]);

    caffe::Caffe::set_random_seed(random_seed);
    printf("Using random seed: %d\n", random_seed);

#ifdef CPU_ONLY
    printf("Setting up Caffe in CPU mode\n");
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    printf("Setting up Caffe in GPU mode with ID: %d\n", gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(gpu_id);
#endif

    // Load the image data.
    LoaderCelebA image_loader(videos_folder_celeba, annotations_folder_celeba);
    const std::vector<CelebAAnnotation>&train_images = image_loader.get_images();
    printf("Total training images: %zu\n", train_images.size());


    // Create an ExampleGenerator to generate training examples.
    ExampleGenerator example_generator(lambda_shift, lambda_scale,
                                       min_scale, max_scale);

    // Set up network.
    RegressorTrain regressor_train(train_proto, caffe_model,
                                   gpu_id, solver_file);

    // Set up trainer.
    TrackerTrainer tracker_trainer(&example_generator, &regressor_train);

    // Train tracker.
    while (tracker_trainer.get_num_batches() < kNumBatches) {
        // Train on an image example.
        train_image(image_loader, train_images, &tracker_trainer);

    }

    return 0;
}

