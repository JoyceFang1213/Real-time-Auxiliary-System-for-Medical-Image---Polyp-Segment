#include "opencv2/opencv.hpp"
#include <iostream>
#include <glog/logging.h>
#include <thread> //
#include <time.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xir/graph/graph.hpp>
#include <cstring>
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vitis/ai/collection_helper.hpp"

template <class T>
class SafeQueue
{
public:
    SafeQueue() : q(), m(), c() {}
    ~SafeQueue() {}
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c.notify_one();
    }

    T dequeue(void) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) c.wait(lock);
        T val = q.front();
        q.pop();
        return val;
    }

    int size(void) { return q.size(); }

private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
};


static void preprocess_image( cv::VideoCapture& cap, 
                              SafeQueue<cv::Mat> &ori_image_queue, 
                              SafeQueue<std::vector<int8_t>> &image_queue) {
    cv::Mat image;
    int testsize = 352; 
    int fix_scale = 16;
    while(1){
        if(ori_image_queue.size()<5){
            cap >> image;
            ori_image_queue.enqueue(image);
            if (image.empty()) break;
            cv::Mat resized_image;
            cv::resize(image, resized_image, cv::Size(testsize, testsize));
            //setImageRGB
            float mean[3] = {103.53f, 116.28f, 123.675f};
            float scales[3] = {0.017429f, 0.017507f, 0.01712475f};
            vector<int8_t> data_tmp;
            data_tmp.reserve(3*testsize*testsize);
            for (auto row = 0; row < resized_image.rows; row++) {
                for (auto col = 0; col < resized_image.cols; col++) {
                    auto v = resized_image.at<cv::Vec3b>(row, col);
                    // substract mean value and times scale;
                    auto B = (float) v[0];
                    auto G = (float) v[1];
                    auto R = (float) v[2];
                    auto nB = (B - mean[0]) * scales[0] * fix_scale;
                    auto nG = (G - mean[1]) * scales[1] * fix_scale;
                    auto nR = (R - mean[2]) * scales[2] * fix_scale;
                    nB = std::max(std::min(nB, 127.0f), -128.0f);
                    nG = std::max(std::min(nG, 127.0f), -128.0f);
                    nR = std::max(std::min(nR, 127.0f), -128.0f);
                    data_tmp.push_back((int) (nR));
                    data_tmp.push_back((int) (nG));
                    data_tmp.push_back((int) (nB));
                }
            }
            image_queue.enqueue(data_tmp);
        } else sleep(0.01);
    }
}

/**** fix_point to scale for input tensor ****/
static float get_input_scale(const xir::Tensor *tensor) {
    int fixpos = tensor->template get_attr<int>("fix_point");
    return std::exp2f(1.0f * (float) fixpos);
}

/**** fix_point to scale for output tensor ****/
static float get_output_scale(const xir::Tensor *tensor) {
    int fixpos = tensor->template get_attr<int>("fix_point");
    return std::exp2f(-1.0f * (float) fixpos);
}

static void post_processing(SafeQueue<vector<float>> &out_image_queue,
                            SafeQueue<cv::Mat> &ori_image_queue,
                            SafeQueue<cv::Mat> &demo_video){

    int testsize = 352;
    int count = 0;
    float time;
    while(1) {
        count++;
        //std::cout<<"post\n";
        clock_t start = clock();
        cv::Mat left = ori_image_queue.dequeue();
        vector<float> out = out_image_queue.dequeue();
        uint8_t img[testsize * testsize];
        for (int i = 0; i < testsize*testsize; i++) img[i] = (out.at(i) > -2.) ? 255 : 0;

        /**** Paste mask ****/
        cv::Mat right_output;
        float rate;
        if(left.rows>1000 || left.cols>1000) rate = 3;
        else if(left.rows>500 || left.cols>500) rate = 1.5; 

        int row = int(left.rows/rate);
        int col = int(left.cols/rate);
        cv::resize(left, left, cv::Size(col, row));
        left.copyTo(right_output);
        cv::Mat mask = cv::Mat(testsize, testsize, CV_8UC1, img);
        cv::resize(mask, mask, cv::Size(col, row));

        for (int i = 0; i < row; ++i)
            for (int j = 0; j < col; ++j)
                if (mask.at<uchar>(i,j) == 255)
                    right_output.at<cv::Vec3b>(i, j)[0] = right_output.at<cv::Vec3b>(i, j)[1] * 0.6 + 255 * 0.4;

        /**** Display the resulting frame ****/
        cv::Mat out_frame;
        cv::hconcat(left, right_output, out_frame);
        cv::resize(out_frame, out_frame, cv::Size(col*2, row));
        /*clock_t end = clock();

        if(count%20 == 0) time = 1. / ((double) (end-start) / CLOCKS_PER_SEC);

        std::ostringstream streamObj;
        streamObj << std::setprecision(4);
        streamObj << time;
        std::string strtime = streamObj.str();
        std::string text = "fps:" + strtime;
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.8;
        cv::Point origin;
           origin.x = out_frame.cols / 2 - 60;
           origin.y = out_frame.rows - 30;
        cv::putText(out_frame, text, origin, font_face, font_scale, cv::Scalar(0, 0, 255));*/

        demo_video.enqueue(out_frame);
        //imshow(video_name, out_frame);
        /**** Press  ESC on keyboard to exit ****/
        //if(cv::waitKey(25) == 27) return;
    }
}

static void show_video( vector<std::string> &video_name,
                        vector<SafeQueue<cv::Mat>> &demo_image){
    int v_num = video_name.size();
    vector<clock_t> ins_time();
    vector<float> ttime(v_num);
    int count=0;

    /*for(int i=0; i<v_num; i++) {
        start[i] = 1;
	end[i] = 10;
    }*/
    //clock_t tlast = clock();
    //clock_t tcurrent = clock();
 
    vector<double> tlast(v_num);
    vector<double> tcurrent(v_num);
    //vector<int> count(v_num);   
    while(1){
	count++;
        for(int i=0; i<v_num; i++){
            tcurrent[i] = clock();
	    cv::Mat out = demo_image[i].dequeue();

            if(count==20) {
                ttime[i] = 1. / ((double) (tcurrent[i]-tlast[i]) / CLOCKS_PER_SEC);
		//count = 0;

                //std::cout << i << " " << count << " " << tcurrent[i] << " " << tlast[i] << " " << ttime[i] << std::endl;
            }

	    if(count==20 && i == v_num-1) count = 0;
	    
            tlast[i] = tcurrent[i];
	    //std::cout << i << std::endl;
	    //std::

            std::ostringstream streamObj;
	    streamObj << std::setprecision(4);
	    streamObj << ttime[i];
	    std::string strtime = streamObj.str();
	    std::string text = "fps:" + strtime;
	    int font_face = cv::FONT_HERSHEY_COMPLEX;
	    double font_scale = 0.8;
	    cv::Point origin;
            origin.x = out.cols / 2 - 60;
	    origin.y = out.rows - 30;
            cv::putText(out, text, origin, font_face, font_scale, cv::Scalar(0, 0, 255));
            imshow(video_name[i], out);
            cv::waitKey(10);

        }
    }
}

int main(int argc, char *argv[]){
    if (argc < 3) {
        cout << "usage: " << argv[0]
             << " <xmodel> <video> \n";
        return 0;
    }

    int file_num = argc - 2;
    std::string xmodel_file = std::string(argv[1]);
    vector<std::string> video_name(file_num);
    vector<cv::VideoCapture> video_cap(file_num);
    vector<std::thread> pre_process_thread;
    vector<std::thread> post_process_thread;
    vector<SafeQueue<cv::Mat>> ori_image_queue(file_num);
    vector<SafeQueue<cv::Mat>> copy_image_queue(file_num);
    vector<SafeQueue<std::vector<int8_t>>> image_queue(file_num);
    std::vector<SafeQueue<vector<float>>> output_image_queue(file_num);
    vector<SafeQueue<cv::Mat>> demo_image(file_num);

    /**** Create dpu runner ****/
    for(int i=2; i<argc; i++){
        std::string name = std::string(argv[i]);
        video_name[i-2] = name;
        cv::VideoCapture cap(name);
        int read_fps = 30;
        cap.set(cv::CAP_PROP_FPS, read_fps); 
        if(!cap.isOpened()){
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
        cv::namedWindow(name);
        video_cap[i-2] = cap;
    }

    /**** Create dpu runner ****/
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto root = graph->get_root_subgraph();
    xir::Subgraph *subgraph = nullptr;
    for (auto c : root->children_topological_sort()) {
        CHECK(c->has_attr("device"));
        if (c->get_attr<std::string>("device") == "DPU") {
            subgraph = c;
            break;
        }
    }
    auto attrs = xir::Attrs::create();
    std::unique_ptr <vart::RunnerExt> runner = vart::RunnerExt::create_runner(subgraph, attrs.get());
    /**** Get input & output tensor buffers ****/
    auto input_tensor_buffers = runner->get_inputs();
    auto output_tensor_buffers = runner->get_outputs();
    CHECK_EQ(input_tensor_buffers.size(), 1u) << "only support HarDMESG model";
    CHECK_EQ(output_tensor_buffers.size(), 1u) << "only support HarDMESG model";
	
    /**** Get input_scale & output_scale ****/
    auto input_tensor = input_tensor_buffers[0]->get_tensor();
    auto input_scale = get_input_scale(input_tensor);
    auto output_tensor = output_tensor_buffers[0]->get_tensor();
    auto output_scale = get_output_scale(output_tensor);
    int testsize = 352;    
    int count = 0;
    double time;

    for(int i=0; i<argc-2; i++){
        pre_process_thread.push_back(std::thread( preprocess_image,
                                                  std::ref(video_cap[i]),
                                                  std::ref(ori_image_queue[i]),
                                                  std::ref(image_queue[i])));
    }

    for(int i=0; i<argc-2; i++){
        post_process_thread.push_back(std::thread( post_processing,
                                                   std::ref(output_image_queue[i]),
                                                   std::ref(ori_image_queue[i]),
                                                   std::ref(demo_image[i])));
    }

    std::thread show_image(show_video, std::ref(video_name), std::ref(demo_image));

    while(1) {
        /**** Pre-processing ****/
        for(int i=0; i<file_num; i++) {
            uint64_t data_in = 0u; 
            size_t size_in = 0u;
            std::tie(data_in, size_in) = input_tensor_buffers[0]->data(std::vector < int > {i, 0, 0, 0});
            CHECK_NE(size_in, 0u);
            signed char *data_tmp = (signed char *) data_in;
            signed char *qdata = (signed char *) image_queue.at(i).dequeue().data();
            std::memcpy(data_tmp, qdata, size_in + 1);
        }

        /**** Run DPU ****/
        for (auto& input : input_tensor_buffers) {
            input->sync_for_write(0, input->get_tensor()->get_data_size() /
            input->get_tensor()->get_shape()[0]);
        }
        auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
            auto status = runner->wait((int) v.first, -1);
        CHECK_EQ(status, 0) << "failed to run dpu";
        for (auto& output : output_tensor_buffers) {
            output->sync_for_read(0, output->get_tensor()->get_data_size() /
            output->get_tensor()->get_shape()[0]);
        }
        
        /**** Post-processing ****/
        uint64_t data;
        size_t size;
        for(int i=0; i<file_num; i++){
            std::tie(data, size) = output_tensor_buffers[0]->data(std::vector < int > {i, 0, 0, 0});
            signed char *data_d = (signed char *) data;
            auto output_image = std::vector<float>(size);
            std::transform(data_d, data_d + size, output_image.begin(),
                    [output_scale](signed char v) { return ((float) v) * output_scale; });
            CHECK_NE(size, 0u);
            output_image_queue[i].enqueue(output_image);
        }
    }

    return 0;
}
