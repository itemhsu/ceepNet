all:
	#gcc classification.cpp
	g++ classification.cpp -MMD -MP -pthread -fPIC -DCAFFE_VERSION=1.0.0-rc3 -DCPU_ONLY -DNDEBUG -O2 -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB -isystem /usr/include/python2.7 -isystem /usr/lib/python2.7/dist-packages/numpy/core/include -isystem /usr/local/include -isystem /home/itemhsu/src/c/ssd/.build_release/src -isystem /home/itemhsu/src/c/ssd/src -isystem /home/itemhsu/src/c/ssd/include -isystem  -Wall -Wno-sign-compare -c -o classification.o 
	g++ classification.o -o classification.bin -pthread -fPIC -DCAFFE_VERSION=1.0.0-rc3  -DCPU_ONLY -DNDEBUG -O2 -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB -isystem /usr/include/python2.7 -isystem /usr/lib/python2.7/dist-packages/numpy/core/include -isystem /usr/local/include -isystem .build_release/src -isystem ./src -isystem ./include  -Wall -Wno-sign-compare -lcaffe -L/usr/lib -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64   -L/usr/lib/x86_64-linux-gnu/ -L/usr/local/lib -L/home/itemhsu/src/c/ssd/.build_release/lib  -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lboost_regex -lm -lboost_regex -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lleveldb -lsnappy -llmdb -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lopenblas

clean:
	rm *bin -rf
	rm *o -rf
	rm *d -rf
