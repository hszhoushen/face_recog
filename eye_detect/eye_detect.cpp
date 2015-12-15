//http://m.blog.csdn.net/blog/computerme/38142125/

#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>


using namespace std;
using namespace cv;


void DetectAndDraw(IplImage* img, CascadeClassifier& cascade);

const char * cascadeName = "../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";


int main( )
{
	CascadeClassifier cascade;
    
	cascade.load( cascadeName );
    
	cvNamedWindow( "result", 1 );

	IplImage* iplImg = cvLoadImage("3.jpg");

	DetectAndDraw( iplImg, cascade );

	cvWaitKey(0);

	cvDestroyWindow("result");

	return 0;
}


void DetectAndDraw(IplImage* img, CascadeClassifier& cascade)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] =  { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255)} ;
	IplImage* gray = cvCreateImage(cvGetSize(img),8,1);
	cvCvtColor( img, gray, CV_BGR2GRAY );
	cvEqualizeHist( gray, gray );

	t = (double)cvGetTickCount();
	cascade.detectMultiScale( gray , faces,
		1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		|CV_HAAR_SCALE_IMAGE
		,
		Size(10, 10) );
	t = (double)cvGetTickCount() - t;
	printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
	{
		Point center;
		Scalar color = colors[i%8];
		int radius;
		//center可以作为瞳孔的坐标
		center.x = cvRound(r->x + r->width*0.5);
		center.y = cvRound(r->y + r->height*0.5);
		//radius = (int)(cvRound(r->width + r->height)*0.25);
		radius =2;
		cvCircle( img, center, radius, color, 3, 8, 0 );
		cvShowImage( "result", img );
	}

	cvShowImage( "result", img );
}