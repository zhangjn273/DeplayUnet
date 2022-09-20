#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

class Detector
{
public:
    Detector(string model_path) : m_bIsNetOK(false)
    {
        UNet = cv::dnn::readNetFromONNX(model_path);

        //下面两行在使用CUDA检测时使用
        // UNet.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        // UNet.setPreferableTarget(dnn::DNN_TARGET_CUDA);

        //使用CPU时使用
        UNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        UNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        m_strOutName = UNet.getUnconnectedOutLayersNames()[0];
        m_bIsNetOK = true;
    };

    ~Detector(){};

    int detect(Mat ImgIn, Mat &ImgOut)
    {
        if (!m_bIsNetOK || ImgIn.empty())
            return 1;

        Mat BlobInput;
        dnn::blobFromImage(ImgIn, BlobInput, 1 / 255.0, Size(256, 256), Scalar(0, 0, 0), true);

        UNet.setInput(BlobInput);
        cv::Mat out_tensor = UNet.forward(m_strOutName);

        vector<Mat> outputImgs;
        dnn::imagesFromBlob(out_tensor, outputImgs);
        resize(outputImgs[0], ImgOut, ImgIn.size());
        return 0;
    }

private:
    dnn::Net UNet;
    bool m_bIsNetOK;
    string m_strOutName;
};

int main(int, char **)
{
    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;
    const std::string img_path = "/home/ubuntu/Documents/Code/git/darknet/DetectSmoke/JPEGImages/bing252.jpg";
    const std::string onnx_model_path = "/home/ubuntu/Documents/Code/Python/Unet_toy-main/trynet.onnx";

    Mat ImgIn = imread(img_path);
    Mat ImgOut;
    Detector *detector = new Detector(onnx_model_path);
    detector->detect(ImgIn, ImgOut);
    imshow("out", ImgOut);
    imshow("in", ImgIn);
    waitKey(0);
}
