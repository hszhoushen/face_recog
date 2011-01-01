#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>

using namespace std;
vector<Mat> &images; 
vector<int> &labels;

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

//实现了从trainningdata 目录下直接读取jpg文件作为训练集
bool read_img(vector<Mat> &images, vector<int> &labels)
{
	DIR * dp;
	struct dirent *ent;			//目录的结构体

	FILE * fp;
	char tuozhanname[5]={0};	//".jpg",存文件名后四位值

	string path = "../einfacedata/trainingdata/";
	
	dp = opendir(path);

	//判断路径是否为空
	if(dp == NULL)
    {
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
			images.push_back(imread(path+ent->d_name, 0));
			labels.push_back(0); 
			continue;	
		}	
}

int main()
{
	read_img(images,labels);
}
