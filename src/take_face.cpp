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
cv::Ptr<cv::FaceRecognizer> model = cv::createLBPHFaceRecognizer();//LBP的这个方法在单个人脸验证方面效果最好
//摄像头捕捉
CvCapture* capture;
//分类器变量
CvHaarClassifierCascade* cascade = 0;
CvHaarClassifierCascade* nested_cascade = 0;
//分类器
const char* cascade_name =
    "../data/haarcascades/haarcascade_frontalface_alt.xml";
const char* nested_cascade_name =
    "../data/haarcascade_eye_tree_eyeglasses.xml";
//创建一个内存存储器，来统一管理各种动态对象的内存
CvMemStorage* storage = 0;
//人脸数组
IplImage* resizeRes;//存放检测到的人脸
IplImage* faceGray; //存放检测到的人脸 灰度图像

IplImage *cpy_Frame;

//scale人脸比例
double scale = 1.0;

//人脸置信度
double dConfidence = 0.0;			
int predictedLabel = 1000;

//两个容器images,labels来存放图像数据和对应的标签
vector<Mat> images;
vector<int> labels;
//file_name
char path[50] = {0};
/*2015.12.7_20:29*/
#if 0
/*从字符串的右边截取n个字符*/
/*usage(judge .mp3 file)   */
char * right(char *dst,char *src, int n)
{
    char *p = src;
    char *q = dst;
    int len = strlen(src);
    if(n>len) n = len;
    p +=(len-n);   /*从右边第n个字符开始*/
    while(*(q++) = *(p++));
    return dst;
}

#if 1
//实现了从trainningdata 目录下直接读取jpg文件作为训练集
bool read_img(vector<Mat> &images, vector<int> &labels)
{
    DIR * dp;
	struct dirent *ent;			//目录的结构体

	FILE * fp;
	char tuozhanname[5]={0};	//".jpg",存文件名后四位值

	char path[50] = "../einfacedata/trainingdata/";
	
	dp = opendir(path);

	//判断路径是否为空
	if(dp == NULL){
		printf("opendir failed\n");
		return NULL;
	}
	//修改工作路径到当前路径
	chdir(path);
	//遍历当前路径
	while((ent = readdir(dp)) != NULL)
	{
		//省略 . && .. 的文件
		if(ent->d_name[0] == '.') continue;
		
		right(tuozhanname,ent->d_name,4);	//提取文件名(ent->d_name)的后四位作为拓展名

		//判断拓展名是否为".jpg"
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
    DIR* dir_info;              //目录指针
    struct dirent* dir_entry;   //目录项信息指针
    //打开一个待扫描的目录
    dir_info = opendir("../einfacedata/trainingdata/*.jpg");
    	//判断路径是否为空
	if(dir_info == NULL){
		printf("opendir failed\n");
		return NULL;
	}
    if(dir_info)
    { 
        //打开目录成功
        while ( (dir_entry = readdir(dir_info)) != NULL)
        {     
            //忽略这两个特殊项目, .. &&　．
            if(strcmp(dir_entry->d_name, "..")==0 || strcmp(dir_entry->d_name, ".")==0)
                continue;
            //具体操作。。。
            
        
            images.push_back(imread(path+find.name, 0));
    		labels.push_back(0); 
        } // while

        //使用完毕，关闭目录指针。
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

//读取文件中的图像数据和类别，存入images和labels这两个容器
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
		getline(liness, path, separator);   //遇到分号就结束
		getline(liness, classlabel);        //继续从分号后面开始，遇到换行结束
		if(!path.empty() && !classlabel.empty())
		{
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


//检测并识别人脸，并在每帧图片上写入结果
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
    cvCvtColor( img, gray, CV_BGR2GRAY ); // 彩色RGB图像转为灰度图像 
    cvResize( gray, small_img, CV_INTER_LINEAR );

    cvEqualizeHist( small_img, small_img ); // 直方图均衡化 

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
       
		//t = (double)cvGetTickCount() - t; // 统计检测使用时间 

        //printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
        for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i ); // 将faces数据从CvSeq转为CvRect 
            CvMat small_img_roi;
            CvSeq* nested_objects;
            CvPoint center,recPt1,recPt2;
            CvScalar color = colors[i%8]; // 使用不同颜色绘制各个face，共八种色 
        
			int radius;
           
			center.x = cvRound((r->x + r->width*0.5)*scale); // 找出faces中心 
            center.y = cvRound((r->y + r->height*0.5)*scale);
			
			recPt1.x = cvRound((r->x)*scale);
			recPt1.y = cvRound((r->y)*scale);
			recPt2.x = cvRound((r->x + r->width)*scale);
			recPt2.y = cvRound((r->y + r->height)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale); 
				
			//r:指定区域的范围,small_img_roi存储从原矩阵中提取的区域,small_img源矩阵或者图像
			cvGetSubRect( small_img, &small_img_roi, *r );
			
			IplImage *result;
			CvRect roi;
			roi = *r;
			//与输入img相同的数据结构
			result = cvCreateImage( cvSize(r->width, r->height), img->depth, img->nChannels );

			//设置ROI
			//大多数OpenCV函数都支持ROI，并将它作为一个独立图像进行处理，所有像素坐标都是从ROI的左上角或者左下角(基于图像结构)开始计算的。
			cvSetImageROI(img,roi);
			// 创建子图像
			cvCopy(img,result);
			cvResetImageROI(img);
			
			//生成100*100的人脸图片
			//IplImage *resizeRes;创建100*100的IplImage数据
			CvSize dst_cvsize;
			dst_cvsize.width=(int)(100);
			dst_cvsize.height=(int)(100);
			resizeRes = cvCreateImage(dst_cvsize,result->depth,result->nChannels);

			//将resuult转换为100*100到resizeRes上
			cvResize(result,resizeRes,CV_INTER_NN);

			faceGray = cvCreateImage(cvGetSize(resizeRes), IPL_DEPTH_8U, 1);//创建目标图像	

			//将彩色人脸转换为灰色人脸
			cvCvtColor(resizeRes,faceGray,CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)

			//显示调整后的彩色人脸
            //cvShowImage( "resize", resizeRes );
			cvShowImage( "gray_face", faceGray);

#ifdef pic_test
        sprintf(path, "../collectfacedata/%d.jpg",count+1);

        cvSaveImage(path, faceGray);

        printf("Sava Picture %d.jpg Finished\n\r",count+1); 

        count++;
        usleep(1000*10);
#endif
			//通过对角线上的两个顶点绘制简单、指定粗细或者带填充的矩形
			cvRectangle(img,recPt1,recPt2,color,2, 8,0);
			
			//cvCircle( img, center, radius, color, 3, 8, 0 ); // 从中心位置画圆，圈出脸部区域

//耗时部分

			Mat test = faceGray;
			//images[images.size() - 1] = test;
			//model->train(images, labels);	  把训练移动到按钮回调函数中
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

// 只是检测，并圈出人脸
void detect_and_draw( IplImage* img ) 
{
    static CvScalar colors[] =		//存储圈出人脸线条的颜色
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
	//开辟人脸临时存储变量,单通道
    gray = cvCreateImage( cvSize(img->width,img->height), IPL_DEPTH_8U, 1 );

	//人脸缩小到的尺寸
    small_img = cvCreateImage( cvSize( cvRound (img->width/scale),		    //int cvRound (double value),对一个double型的数进行四舍五入，并返回一个整型数！

    cvRound (img->height/scale)), IPL_DEPTH_8U, 1 );	//scale = 1
	//彩色RGB图像转为灰度图像 
    cvCvtColor( img, gray, CV_BGR2GRAY ); 

	//显示灰度图
	//cvShowImage("gray", gray);

	//双线性插值（默认方法）,缩放源图像到目标图像
    cvResize( gray, small_img, CV_INTER_LINEAR );

	// 直方图均衡化 ， 作用是归一化图像亮度和增强对比度
    cvEqualizeHist( small_img, small_img ); 
	//清除内存存储块，不释放内存
    cvClearMemStorage( storage );

    if( cascade )			//如果加载分类器成功
    {

       // double t = (double)cvGetTickCount(); 
        CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,		//用来检测图像中的目标,haarcascade_frontalface_alt.xml,人脸分类器
                                            1.1, 2, 0
                                            //|CV_HAAR_FIND_BIGGEST_OBJECT
                                            |CV_HAAR_DO_ROUGH_SEARCH
                                            //|CV_HAAR_DO_CANNY_PRUNING
                                            //|CV_HAAR_SCALE_IMAGE
                                            ,
                                            cvSize(200, 200) );		// 检测窗口的最小尺寸
      //  t = (double)cvGetTickCount() - t; // 统计检测使用时间 
        //printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
        for( i = 0; i < (faces ? faces->total : 0); i++ )		//若检测到人脸，则 i < faces->total , 否则， i < 0
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i ); // 将faces数据从CvSeq转为CvRect 

            CvMat small_img_roi;

           // CvSeq* nested_objects;

            CvPoint center,recPt1,recPt2;

            CvScalar color = colors[i%8]; // 使用不同颜色绘制各个face，共八种色 

            int radius;
           
			center.x = cvRound((r->x + r->width*0.5)*scale); // 找出faces中心 
            center.y = cvRound((r->y + r->height*0.5)*scale);
			
			recPt1.x = cvRound((r->x)*scale);
			recPt1.y = cvRound((r->y)*scale);
			recPt2.x = cvRound((r->x + r->width)*scale);
			recPt2.y = cvRound((r->y + r->height)*scale);
            
			radius = cvRound((r->width + r->height)*0.25*scale); 
				
			//作用是从一个图像中提取出来一部分到另一个图像
			cvGetSubRect( small_img, &small_img_roi, *r );
			
			IplImage *result;
			CvRect roi;
			roi = *r;

			//result大小与r有关
			result = cvCreateImage( cvSize(r->width, r->height), img->depth, img->nChannels );
			
			//设置ROI
			cvSetImageROI(img,roi);
			// 创建子图像
			cvCopy(img,result);
			//释放ROI
			cvResetImageROI(img);
			
			//显示result
			//cvShowImage("result", result);

			//IplImage *resizeRes;
			CvSize dst_cvsize;
			dst_cvsize.width=(int)(100);
			dst_cvsize.height=(int)(100);
            
			resizeRes = cvCreateImage(dst_cvsize,result->depth,result->nChannels);

			cvResize(result,resizeRes,CV_INTER_NN);

			faceGray = cvCreateImage(cvGetSize(resizeRes), IPL_DEPTH_8U, 1);//创建目标图像	
			cvCvtColor(resizeRes,faceGray,CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)


           // cvShowImage( "resize", resizeRes );
			//cvShowImage("face_Gray", faceGray);				   // 从中心位置画矩形，圈出脸部区域

			cvRectangle(img,recPt1,recPt2,color,1, 8,0);
			
			//cvCircle( img, center, radius, color, 3, 8, 0 ); // 从中心位置画圆，圈出脸部区域

        }
    }
//    cvShowImage( "result", gray );

    cvShowImage( "result", resizeRes );
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}


#ifdef thread
//线程处理函数，每隔一段时间获取一次图片
void *pthread_fun(void * arg)
{
	
	pthread_detach(pthread_self());	//线程分离函数

    while(1)
    {
        usleep(40*1000);
    }
    
    printf("pthread_exit\n\r");
    
	pthread_exit(0);				//线程退出函数
}
#endif

int main(int argc, char *argv[])
{
    IplImage* pFrame = NULL;
    
    int count = 0;
        
    double dConfidence = 75.0;
    
    // 加载人脸识别分类器 
	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); 
	if( !cascade ) 
    {
        cout<<"无法加载人脸识别分类器文件，请确认！"<<endl;
    }
	 
    labels.clear();
    images.clear();

    //读取你的CSV文件路径
	string fn_csv = string("../einfacedata/at.txt");
	try
	{
		//通过./einfacedata/at.txt这个文件读取里面的训练图像和类别标签
		read_csv(fn_csv, images, labels,';');	
	}

    catch(cv::Exception &e)
	{
		cerr<<"Error opening file "<<fn_csv<<". Reason: "<<e.msg<<endl;
		exit(1);
	}

    //设置阈值
    model->set("threshold", 2100.0);
   
    cout << "after_set_fazhi" << endl;

    // 创建内存存储器 
    storage = cvCreateMemStorage(0); 
    //如果没有读到足够的图片，就退出
    cout << images.size() << ":" << labels.size()<<endl;
    
	if(images.size() <= 2)
	{
		string error_message = "This demo needs at least 2 images to work.";
		CV_Error(CV_StsError, error_message);
	}  
    //进行训练
	model->train(images, labels);
  
    //创建名字为video的窗口
    cvNamedWindow("Video", 1);

    //打开摄像头
    capture = cvCaptureFromCAM(0);
  
#ifdef thread   
    pthread_t tid;	//线程定义
    //创建新线程来处理新的连接
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


