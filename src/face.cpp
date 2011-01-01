#include <stdio.h>  
#include <string.h>  
#include <unistd.h>

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#include <pthread.h>

using namespace cv;

//获取摄像头实时图像
IplImage* m_Frame;

CvCapture* capture;

//线程处理函数
void *pthread_fun(void * arg)
{
	
	pthread_detach(pthread_self());	//线程分离函数


 //   capture = cvCaptureFromCAM(-1);
      capture = cvCreateCameraCapture(0);
//    cvNamedWindow("Video", 0);
	while(1)
    {
 #if 0       
        if(capture)
        {
            m_Frame = cvQueryFrame(capture);
        }
        if(!m_Frame)break;
        
        cvShowImage("Video",m_Frame);
        
        char c = cvWaitKey(33);        
        if(c==27)break;
 #endif     
        printf("ERROR\n\r");
        usleep(1000*50);
	}
    
    printf("pthread_exit\n\r");
    
	pthread_exit(0);				//线程退出函数
}

int main()
{
    pthread_t tid;	//线程定义
   

    //创建新线程来处理新的连接
	pthread_create(&tid, NULL, pthread_fun, NULL);
    
    while(1)
    {
       printf("This is main process\r\n");
       sleep(1);
    }

 //   cvDestroyWindow("Video");
	
  //  cvReleaseCapture(&capture);
  
}
