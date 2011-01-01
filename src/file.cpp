#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>

using namespace std;
vector<Mat> &images; 
vector<int> &labels;

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

//ʵ���˴�trainningdata Ŀ¼��ֱ�Ӷ�ȡjpg�ļ���Ϊѵ����
bool read_img(vector<Mat> &images, vector<int> &labels)
{
	DIR * dp;
	struct dirent *ent;			//Ŀ¼�Ľṹ��

	FILE * fp;
	char tuozhanname[5]={0};	//".jpg",���ļ�������λֵ

	string path = "../einfacedata/trainingdata/";
	
	dp = opendir(path);

	//�ж�·���Ƿ�Ϊ��
	if(dp == NULL)
    {
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
			images.push_back(imread(path+ent->d_name, 0));
			labels.push_back(0); 
			continue;	
		}	
}

int main()
{
	read_img(images,labels);
}
