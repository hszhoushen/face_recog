#include <stdio.h>  
#include <string.h>  
#include <unistd.h>

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#include <pthread.h>

using namespace cv;

//��ȡ����ͷʵʱͼ��
IplImage* m_Frame;

CvCapture* capture;

//�̴߳�����
void *pthread_fun(void * arg)
{
	
	pthread_detach(pthread_self());	//�̷߳��뺯��


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
    
	pthread_exit(0);				//�߳��˳�����
}

int main()
{
    pthread_t tid;	//�̶߳���
   

    //�������߳��������µ�����
	pthread_create(&tid, NULL, pthread_fun, NULL);
    
    while(1)
    {
       printf("This is main process\r\n");
       sleep(1);
    }

 //   cvDestroyWindow("Video");
	
  //  cvReleaseCapture(&capture);
  
}
