#include <stdio.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <unistd.h>
#include <string.h>



//2015.12.7_15:36_add
#include <opencv2/contrib/contrib.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <string>
//2015.12.7_19:19
#include<dirent.h>

//2015.12.7_20:32
//#include <io.h> 
#include <fstream>
#include <sstream>

//namespace
using namespace cv;
using namespace std;


//test_define

#define pic_test
//model
cv::Ptr<cv::FaceRecognizer> model = cv::createLBPHFaceRecognizer();//LBP����������ڵ���������֤����Ч�����
//����ͷ��׽
CvCapture* capture;
//����������
CvHaarClassifierCascade* cascade = 0;
CvHaarClassifierCascade* nested_cascade = 0;
//������
const char* cascade_name =
    "../data/haarcascades/haarcascade_frontalface_alt.xml";
const char* nested_cascade_name =
    "../data/haarcascade_eye_tree_eyeglasses.xml";
//����һ���ڴ�洢������ͳһ������ֶ�̬������ڴ�
CvMemStorage* storage = 0;
//��������
IplImage* resizeRes;//��ż�⵽������
IplImage* faceGray; //��ż�⵽������ �Ҷ�ͼ��

IplImage *cpy_Frame;

//scale��������
double scale = 1.0;

//�������Ŷ�
double dConfidence = 0.0;			
int predictedLabel = 1000;

//��������images,labels�����ͼ�����ݺͶ�Ӧ�ı�ǩ
vector<Mat> images;
vector<int> labels;
//file_name
char path[50] = {0};
/*2015.12.7_20:29*/
#if 0
/*���ַ������ұ߽�ȡn���ַ�*/
/*usage(judge .mp3 file)   */
char * right(char *dst,char *src, int n)
{
    char *p = src;
    char *q = dst;
    int len = strlen(src);
    if(n>len) n = len;
    p +=(len-n);   /*���ұߵ�n���ַ���ʼ*/
    while(*(q++) = *(p++));
    return dst;
}

#if 1
//ʵ���˴�trainningdata Ŀ¼��ֱ�Ӷ�ȡjpg�ļ���Ϊѵ����
bool read_img(vector<Mat> &images, vector<int> &labels)
{
    DIR * dp;
	struct dirent *ent;			//Ŀ¼�Ľṹ��

	FILE * fp;
	char tuozhanname[5]={0};	//".jpg",���ļ�������λֵ

	char path[50] = "../einfacedata/trainingdata/";
	
	dp = opendir(path);

	//�ж�·���Ƿ�Ϊ��
	if(dp == NULL){
		printf("opendir failed\n");
		return NULL;
	}
	//�޸Ĺ���·������ǰ·��
	chdir(path);
	//������ǰ·��
	while((ent = readdir(dp)) != NULL)
	{
		//ʡ�� . && .. ���ļ�
		if(ent->d_name[0] == '.') continue;
		
		right(tuozhanname,ent->d_name,4);	//��ȡ�ļ���(ent->d_name)�ĺ���λ��Ϊ��չ��

		//�ж���չ���Ƿ�Ϊ".jpg"
		if(strcmp(tuozhanname,".jpg") != 0) 
		{             
			continue;	
		}
        strcat(path, ent->d_name);
		images.push_back(imread(path, 0));
		labels.push_back(0);
	}
    closedir(dp);
#if 0
    DIR* dir_info;              //Ŀ¼ָ��
    struct dirent* dir_entry;   //Ŀ¼����Ϣָ��
    //��һ����ɨ���Ŀ¼
    dir_info = opendir("../einfacedata/trainingdata/*.jpg");
    	//�ж�·���Ƿ�Ϊ��
	if(dir_info == NULL){
		printf("opendir failed\n");
		return NULL;
	}
    if(dir_info)
    { 
        //��Ŀ¼�ɹ�
        while ( (dir_entry = readdir(dir_info)) != NULL)
        {     
            //����������������Ŀ, .. &&����
            if(strcmp(dir_entry->d_name, "..")==0 || strcmp(dir_entry->d_name, ".")==0)
                continue;
            //�������������
            
        
            images.push_back(imread(path+find.name, 0));
    		labels.push_back(0); 
        } // while

        //ʹ����ϣ��ر�Ŀ¼ָ�롣
        closedir(dir_info);
    }
#endif
#if 0
	long file;  
    struct finddata_t find; 
	
	string path = "..//einfacedata//trainingdata/";
	char filepath[60];
    
    if((file=findfirst("..//einfacedata//trainingdata/*.jpg", &find))==-1L) 
	{  
		cout<<"Cannot find the dir"<<endl;
        return false;  
    }  
	int i = 0;
    images.push_back(imread(path+find.name, 0));
	labels.push_back(0); 
    
    while(findnext(file, &find)==0)  
    {  
		//_cprintf("%s\n", path+find.name);
		//_cprintf("%d\n", i++);
		images.push_back(imread(path+find.name, 0));
		labels.push_back(0); 
    }  
    
    findclose(file);
 #endif   
	return true;
}
#endif



void train()
{
   
	images.clear();
	labels.clear();
		       
	dConfidence = 70;	
	
	if(!read_img(images, labels))
	{
		cout<<"Error in reading images!"<<endl;
		images.clear();
		labels.clear();
		return;
	}
    
	model->set("threshold", dConfidence);
	
	
	if(images.size() <= 2)
	{
		cout<<"This demo needs at least 2 image!"<<endl;
		return;
	}

	//training
	model->train(images, labels);

}
#endif

//��ȡ�ļ��е�ͼ�����ݺ���𣬴���images��labels����������
void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator)
{
	std::ifstream file(filename.c_str(), ifstream::in);
	if(!file)
	{
		string error_message = "No valid input file was given.";
		CV_Error(CV_StsBadArg, error_message);
	}

	string line, path, classlabel;
	while(getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);   //�����ֺžͽ���
		getline(liness, classlabel);        //�����ӷֺź��濪ʼ���������н���
		if(!path.empty() && !classlabel.empty())
		{
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


//��Ⲣʶ������������ÿ֡ͼƬ��д����
void recog_and_draw( IplImage* img )
{
    static int count = 0;
    static CvScalar colors[] = 
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}}
    };
    IplImage *gray, *small_img;

    int i, j;
    gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );

    small_img = cvCreateImage( cvSize( cvRound (img->width/scale),
                         cvRound (img->height/scale)), 8, 1 );
    cvCvtColor( img, gray, CV_BGR2GRAY ); // ��ɫRGBͼ��תΪ�Ҷ�ͼ�� 
    cvResize( gray, small_img, CV_INTER_LINEAR );

    cvEqualizeHist( small_img, small_img ); // ֱ��ͼ���⻯ 

    cvClearMemStorage( storage );

    if( cascade )
    {
        //double t = (double)cvGetTickCount(); 

        CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,
                                            1.1, 2, 0
                                            //|CV_HAAR_FIND_BIGGEST_OBJECT
                                            //|CV_HAAR_DO_ROUGH_SEARCH
                                            |CV_HAAR_DO_CANNY_PRUNING
                                            //|CV_HAAR_SCALE_IMAGE
                                            ,
                                            cvSize(200, 200) );
       
		//t = (double)cvGetTickCount() - t; // ͳ�Ƽ��ʹ��ʱ�� 

        //printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
        for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i ); // ��faces���ݴ�CvSeqתΪCvRect 
            CvMat small_img_roi;
            CvSeq* nested_objects;
            CvPoint center,recPt1,recPt2;
            CvScalar color = colors[i%8]; // ʹ�ò�ͬ��ɫ���Ƹ���face��������ɫ 
        
			int radius;
           
			center.x = cvRound((r->x + r->width*0.5)*scale); // �ҳ�faces���� 
            center.y = cvRound((r->y + r->height*0.5)*scale);
			
			recPt1.x = cvRound((r->x)*scale);
			recPt1.y = cvRound((r->y)*scale);
			recPt2.x = cvRound((r->x + r->width)*scale);
			recPt2.y = cvRound((r->y + r->height)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale); 
				
			//r:ָ������ķ�Χ,small_img_roi�洢��ԭ��������ȡ������,small_imgԴ�������ͼ��
			cvGetSubRect( small_img, &small_img_roi, *r );
			
			IplImage *result;
			CvRect roi;
			roi = *r;
			//������img��ͬ�����ݽṹ
			result = cvCreateImage( cvSize(r->width, r->height), img->depth, img->nChannels );

			//����ROI
			//�����OpenCV������֧��ROI����������Ϊһ������ͼ����д��������������궼�Ǵ�ROI�����Ͻǻ������½�(����ͼ��ṹ)��ʼ����ġ�
			cvSetImageROI(img,roi);
			// ������ͼ��
			cvCopy(img,result);
			cvResetImageROI(img);
			
			//����100*100������ͼƬ
			//IplImage *resizeRes;����100*100��IplImage����
			CvSize dst_cvsize;
			dst_cvsize.width=(int)(100);
			dst_cvsize.height=(int)(100);
			resizeRes = cvCreateImage(dst_cvsize,result->depth,result->nChannels);

			//��resuultת��Ϊ100*100��resizeRes��
			cvResize(result,resizeRes,CV_INTER_NN);

			faceGray = cvCreateImage(cvGetSize(resizeRes), IPL_DEPTH_8U, 1);//����Ŀ��ͼ��	

			//����ɫ����ת��Ϊ��ɫ����
			cvCvtColor(resizeRes,faceGray,CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)

			//��ʾ������Ĳ�ɫ����
            //cvShowImage( "resize", resizeRes );
			cvShowImage( "gray_face", faceGray);

#ifdef pic_test
        sprintf(path, "../collectfacedata/%d.jpg",count+1);

        cvSaveImage(path, faceGray);

        printf("Sava Picture %d.jpg Finished\n\r",count+1); 

        count++;
        usleep(1000*10);
#endif
			//ͨ���Խ����ϵ�����������Ƽ򵥡�ָ����ϸ���ߴ����ľ���
			cvRectangle(img,recPt1,recPt2,color,2, 8,0);
			
			//cvCircle( img, center, radius, color, 3, 8, 0 ); // ������λ�û�Բ��Ȧ����������

//��ʱ����

			Mat test = faceGray;
			//images[images.size() - 1] = test;
			//model->train(images, labels);	  ��ѵ���ƶ�����ť�ص�������
			//predictedLabel = model->predict(test);
			double predicted_confidence = 0.0;
			model->predict(test, predictedLabel,predicted_confidence);
			
			
			stringstream strStream;
			strStream << predicted_confidence;
			string ss = strStream.str(); 

			//cvText(img, ss.c_str(), r->x+r->width*0.5, r->y);
			//cout<<ss.c_str()<<endl;


            cout << "person is: " << predictedLabel << endl;
            
			if(predictedLabel == 3){
				//cvText(img, "Result:YES", 0, 30); 
				cout << "Yes" << endl;
			}
			else{
				//cvText(img, "Result:NO", 0, 30);  
   				cout << "No" << endl;
			}

			//cout << "predict:"<<model->predict(test) << endl;
			//cout << "predict:"<< predictedLabel << "\nconfidence:" << predicted_confidence << endl;
        }
    }
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}

// ֻ�Ǽ�⣬��Ȧ������
void detect_and_draw( IplImage* img ) 
{
    static CvScalar colors[] =		//�洢Ȧ��������������ɫ
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}}
    };
    IplImage *gray, *small_img;
    int i, j;
	//����������ʱ�洢����,��ͨ��
    gray = cvCreateImage( cvSize(img->width,img->height), IPL_DEPTH_8U, 1 );

	//������С���ĳߴ�
    small_img = cvCreateImage( cvSize( cvRound (img->width/scale),		    //int cvRound (double value),��һ��double�͵��������������룬������һ����������

    cvRound (img->height/scale)), IPL_DEPTH_8U, 1 );	//scale = 1
	//��ɫRGBͼ��תΪ�Ҷ�ͼ�� 
    cvCvtColor( img, gray, CV_BGR2GRAY ); 

	//��ʾ�Ҷ�ͼ
	//cvShowImage("gray", gray);

	//˫���Բ�ֵ��Ĭ�Ϸ�����,����Դͼ��Ŀ��ͼ��
    cvResize( gray, small_img, CV_INTER_LINEAR );

	// ֱ��ͼ���⻯ �� �����ǹ�һ��ͼ�����Ⱥ���ǿ�Աȶ�
    cvEqualizeHist( small_img, small_img ); 
	//����ڴ�洢�飬���ͷ��ڴ�
    cvClearMemStorage( storage );

    if( cascade )			//������ط������ɹ�
    {

       // double t = (double)cvGetTickCount(); 
        CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,		//�������ͼ���е�Ŀ��,haarcascade_frontalface_alt.xml,����������
                                            1.1, 2, 0
                                            //|CV_HAAR_FIND_BIGGEST_OBJECT
                                            |CV_HAAR_DO_ROUGH_SEARCH
                                            //|CV_HAAR_DO_CANNY_PRUNING
                                            //|CV_HAAR_SCALE_IMAGE
                                            ,
                                            cvSize(200, 200) );		// ��ⴰ�ڵ���С�ߴ�
      //  t = (double)cvGetTickCount() - t; // ͳ�Ƽ��ʹ��ʱ�� 
        //printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
        for( i = 0; i < (faces ? faces->total : 0); i++ )		//����⵽�������� i < faces->total , ���� i < 0
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i ); // ��faces���ݴ�CvSeqתΪCvRect 

            CvMat small_img_roi;

           // CvSeq* nested_objects;

            CvPoint center,recPt1,recPt2;

            CvScalar color = colors[i%8]; // ʹ�ò�ͬ��ɫ���Ƹ���face��������ɫ 

            int radius;
           
			center.x = cvRound((r->x + r->width*0.5)*scale); // �ҳ�faces���� 
            center.y = cvRound((r->y + r->height*0.5)*scale);
			
			recPt1.x = cvRound((r->x)*scale);
			recPt1.y = cvRound((r->y)*scale);
			recPt2.x = cvRound((r->x + r->width)*scale);
			recPt2.y = cvRound((r->y + r->height)*scale);
            
			radius = cvRound((r->width + r->height)*0.25*scale); 
				
			//�����Ǵ�һ��ͼ������ȡ����һ���ֵ���һ��ͼ��
			cvGetSubRect( small_img, &small_img_roi, *r );
			
			IplImage *result;
			CvRect roi;
			roi = *r;

			//result��С��r�й�
			result = cvCreateImage( cvSize(r->width, r->height), img->depth, img->nChannels );
			
			//����ROI
			cvSetImageROI(img,roi);
			// ������ͼ��
			cvCopy(img,result);
			//�ͷ�ROI
			cvResetImageROI(img);
			
			//��ʾresult
			//cvShowImage("result", result);

			//IplImage *resizeRes;
			CvSize dst_cvsize;
			dst_cvsize.width=(int)(100);
			dst_cvsize.height=(int)(100);
            
			resizeRes = cvCreateImage(dst_cvsize,result->depth,result->nChannels);

			cvResize(result,resizeRes,CV_INTER_NN);

			faceGray = cvCreateImage(cvGetSize(resizeRes), IPL_DEPTH_8U, 1);//����Ŀ��ͼ��	
			cvCvtColor(resizeRes,faceGray,CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)


           // cvShowImage( "resize", resizeRes );
			//cvShowImage("face_Gray", faceGray);				   // ������λ�û����Σ�Ȧ����������

			cvRectangle(img,recPt1,recPt2,color,1, 8,0);
			
			//cvCircle( img, center, radius, color, 3, 8, 0 ); // ������λ�û�Բ��Ȧ����������

        }
    }
//    cvShowImage( "result", gray );

    cvShowImage( "result", resizeRes );
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}


#ifdef thread
//�̴߳�������ÿ��һ��ʱ���ȡһ��ͼƬ
void *pthread_fun(void * arg)
{
	
	pthread_detach(pthread_self());	//�̷߳��뺯��

    while(1)
    {
        usleep(40*1000);
    }
    
    printf("pthread_exit\n\r");
    
	pthread_exit(0);				//�߳��˳�����
}
#endif

int main(int argc, char *argv[])
{
    IplImage* pFrame = NULL;
    
    int count = 0;
        
    double dConfidence = 75.0;
    
    // ��������ʶ������� 
	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); 
	if( !cascade ) 
    {
        cout<<"�޷���������ʶ��������ļ�����ȷ�ϣ�"<<endl;
    }
	 
    labels.clear();
    images.clear();

    //��ȡ���CSV�ļ�·��
	string fn_csv = string("../einfacedata/at.txt");
	try
	{
		//ͨ��./einfacedata/at.txt����ļ���ȡ�����ѵ��ͼ�������ǩ
		read_csv(fn_csv, images, labels,';');	
	}

    catch(cv::Exception &e)
	{
		cerr<<"Error opening file "<<fn_csv<<". Reason: "<<e.msg<<endl;
		exit(1);
	}

    //������ֵ
    model->set("threshold", 2100.0);
   
    cout << "after_set_fazhi" << endl;

    // �����ڴ�洢�� 
    storage = cvCreateMemStorage(0); 
    //���û�ж����㹻��ͼƬ�����˳�
    cout << images.size() << ":" << labels.size()<<endl;
    
	if(images.size() <= 2)
	{
		string error_message = "This demo needs at least 2 images to work.";
		CV_Error(CV_StsError, error_message);
	}  
    //����ѵ��
	model->train(images, labels);
  
    //��������Ϊvideo�Ĵ���
    cvNamedWindow("Video", 1);

    //������ͷ
    capture = cvCaptureFromCAM(0);
  
#ifdef thread   
    pthread_t tid;	//�̶߳���
    //�������߳��������µ�����
    pthread_create(&tid, NULL, pthread_fun, NULL);
#endif

    while(1)
    {  
        pFrame=cvQueryFrame(capture);

        if(!pFrame)break;

        recog_and_draw(pFrame);

        cvShowImage("Video",pFrame);


        char c = cvWaitKey(33);

        if(c==27)break;      
	}

    cvReleaseImage(&pFrame);  
    cvReleaseCapture(&capture);
    cvDestroyWindow("Video");
    
    return 0;
}

#ifdef pic

    for(count = 0; count< 20; count++)
    {
        sprintf(path, "../collectfacedata/%d.jpg",count+1);

        cvSaveImage(path, cpy_Frame);

        printf("Sava Picture %d.jpg Finished\n\r",count+1);
        sleep(1);
    }
    
#endif


