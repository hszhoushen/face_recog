#makefile example
CC=g++
CFLAGS=
CFLAGS+=-I./\
		-L./
LFLAGS=-lpthread -lm -lrt\
		-lopencv_nonfree -lopencv_objdetect  -lopencv_features2d\
		-lopencv_legacy  -lopencv_imgproc  -lopencv_core\
		-lopencv_highgui -lopencv_contrib -lopencv_flann\
		-lopencv_gpu -lopencv_imgproc -lopencv_photo\
		-lopencv_stitching -lopencv_ts -lopencv_videostab\
		-lopencv_ml -lopencv_gpu -lopencv_calib3d\
		-lopencv_contrib -lopencv_photo -lopencv_stitching\
		-lopencv_superres -lopencv_video

SRCS=eye_detect.cpp

OBJS=$(SRCS:.cpp=.o)
EXEC=a.out

all:$(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LFLAGS)

clean:
	rm -rf $(EXEC) $(OBJS)
