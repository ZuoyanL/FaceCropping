//
//  main.cpp
//  ImagePreprocess
//
//  Created by 小小延 on 2019/12/14.
//  Copyright © 2019年 Forrest. All rights reserved.
//


#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pthread.h>
#include <assert.h>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#ifdef __APPLE__
#include <sys/uio.h>
#else
#include <sys/io.h>
#endif
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/opencv.h>

using namespace std;
using namespace cv;
//using namespace dlib;


// 互斥访问路径，一个线程处理一个视频。
pthread_mutex_t mutex_queue;
std::queue<string> ALL_PATHS;
int NumbersOfVideos = 0;
int current_video = 0;
string model_path = "/Users/xiaoxiaoyan/Desktop/ImagePreprocess/haarcascade_frontalface_alt.xml";




/****文件名处理****/
//读取所有文件中的mp4/avi的路径。
queue<string> getFiles(string cate_dir, string mode1 = ".mp4", string mode2 = ".avi");
/********END文件名处理************/


/****视频预处理****/
// Implement crop frames from a giant number of videos.
/***
 frame: 对此帧图像进行人脸检测、Crop操作。
 ***/
void detectAndDisplay(Mat frame);
/***
 path: 视频路径
 face_cascade: 级联检测器
 skip: 隔skip帧取一张图片
 ***/
void video2imgs(string path, CascadeClassifier face_cascade, int skip=3);
/****END视频预处理****/

/*********多线程加速***********/
#define NUM_THREADS 3

struct ParamThread
{
    int  thread_id;
    CascadeClassifier face_cascade;
    string path;
    int skip;
};
void *threadProcess(void *args)
{
    pthread_t myid = pthread_self();
    ParamThread *para = (ParamThread*) args;
    bool over = false;
    while(!over)
    {
        pthread_mutex_lock(&mutex_queue);
        if(ALL_PATHS.empty())
        {
            over = true;
            pthread_mutex_unlock(&mutex_queue);
            break;
        }
        string path = ALL_PATHS.front();
        printf("Thread %d handle %s\n", para->thread_id, path.c_str());
        ALL_PATHS.pop();
        pthread_mutex_unlock(&mutex_queue);
        video2imgs(path, para->face_cascade, para->skip);
    }
    pthread_exit(NULL);
};
/*********END多线程加速***********/

int main(int argc, const char * argv[])
{
    ALL_PATHS = getFiles("/Users/xiaoxiaoyan/Desktop/TEST");
    pthread_mutex_init(&mutex_queue, NULL);
    cout<<ALL_PATHS.size();
    
    pthread_t threads[NUM_THREADS];
    struct ParamThread td[NUM_THREADS];
    
    pthread_attr_t attr;
    void *status;
    // 初始化并设置线程为可连接的（joinable）
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    timeval t_start, t_end;
    
    int rc;
    int i;
    int skip = 1;
    gettimeofday( &t_start, NULL);
    for( i=0; i < NUM_THREADS; i++ ){
        cout <<"main() : creating thread, " << i << endl;
        td[i].thread_id = i;
        CascadeClassifier face_cascade;
        if (!face_cascade.load(model_path)){
            printf("--(!)Error loading\n");
            return (-1);
        }
        td[i].face_cascade = face_cascade;
        td[i].skip = skip;
        rc = pthread_create(&threads[i], NULL,
                            threadProcess, (void *)&td[i]);
        if (rc){
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }
    
    pthread_attr_destroy(&attr);
    for( i=0; i < NUM_THREADS; i++ ){
        rc = pthread_join(threads[i], &status);
        if (rc){
            cout << "Error:unable to join," << rc << endl;
            exit(-1);
        }
        cout << "Main: completed thread id :" << i ;
        cout << "  exiting with status :" << status << endl;
    }
    gettimeofday( &t_end, NULL);
    double delta_t = (t_end.tv_sec-t_start.tv_sec) +
    (t_end.tv_usec-t_start.tv_usec)/1000000.0;
    cout << "all time : " << delta_t  << "s" << endl;
    cout << "Main: program exiting." << endl;
    pthread_exit(NULL);
    
//    exit(0);
}
/**********************/
queue<string> getFiles(string cate_dir, string mode1, string mode2)
{
    queue<string> files;//存放文件名
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    if ((dir=opendir(cate_dir.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }
    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
        continue;
        else if(ptr->d_type == 8)
        {///file
            string s = ptr->d_name;
            string::size_type idx1 =s.find(mode1);
            string::size_type idx2 =s.find(mode2);
            if (idx1 != string::npos || idx2 != string::npos)
            {
                files.push(cate_dir+"/"+ptr->d_name);
            }
        }
        else if(ptr->d_type == 10)    ///link file
        continue;
        else if(ptr->d_type == 4)    ///dir
        {
            files.push(cate_dir+"/"+ptr->d_name);
        }
    }
    closedir(dir);
    return files;
}

/**********************/

/**********************/
void detectAndDisplay(Mat frame, CascadeClassifier face_cascade, string filename)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;
    float scale = 1.2;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    
    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.5, 2, 0 | CASCADE_SCALE_IMAGE, Size(int(30*scale), int(30*scale)));
    
    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;
    
    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element
    
    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element
    
    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
    
    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);
        
        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)
        
        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);
        
        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element
        
        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }
        
        crop = frame(roi_b);
//        resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
//        cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale
        
        // Form a filename
//        filename = "";
//        stringstream ssfn;
//        ssfn << filenumber << ".jpg";
//        filename = ssfn.str();
//        filenumber++;
        
//        imwrite(filename, gray);
        
        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
//        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }
    
    // Show image
    sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
    text = sstm.str();
    
//    putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
//    imshow("original", frame);
//    waitKey(200);
    if (!crop.empty())
    {
        imwrite(filename, crop);
//        cout<<"writing..."<<filename<<endl;
//        imshow("detected", crop);
//        waitKey(1000);
    }
    else
    destroyWindow("detected");
}

//void preprocess_images(std::queue<Mat> imgs, std::string output_path, CascadeClassifier face_cascade, int skip)
//{
//}
// from video to images
void video2imgs(string path, CascadeClassifier face_cascade, int skip)
{
    VideoCapture capture;
    Mat frame;
    
    string output_path = path;
    int frame_number = 1;
    string::iterator ite;
    ite = output_path.begin();
    for (size_t i = 0; i < output_path.size(); i++){
        if(*ite == '.')
        break;
    }
    size_t index = 0;
    index = output_path.find(".mp4", index);
    if (index == std::string::npos)
        index = output_path.find(".avi", index);
    
    /* Make the replacement. */
    output_path.replace(index, 4, "_");
    Mat image;
    
    std::queue<Mat> frames;
    frame = capture.open(path);
    if(!capture.isOpened())
    {
        cout<<path<<endl;
        printf(" can not open ...\n");
        exit(0);
    }
    while (capture.read(frame))
    {
        if(frame_number % skip == 0)
        {
            string p = "/" + output_path + to_string(frame_number) + ".jpg";
            detectAndDisplay(frame, face_cascade, p);
        }
        frame_number ++;
    }
    capture.release();
}

/**********************/
