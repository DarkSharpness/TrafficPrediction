EXE_PATH = __exe__
PRE_PATH = preprocess

build:
	make init
	make compile
	make run

init:
	mkdir -p $(EXE_PATH)

run:
	./$(EXE_PATH)/train
	./$(EXE_PATH)/pred
	./$(EXE_PATH)/cleaner
	./$(EXE_PATH)/flatten
	./$(EXE_PATH)/spliter

compile:
	g++ $(PRE_PATH)/train.cpp   -o $(EXE_PATH)/train   -std=c++20 -O3
	g++ $(PRE_PATH)/spliter.cpp -o $(EXE_PATH)/spliter -std=c++20 -O3
	g++ $(PRE_PATH)/pred.cpp    -o $(EXE_PATH)/pred    -std=c++20 -O3
	g++ $(PRE_PATH)/cleaner.cpp -o $(EXE_PATH)/cleaner -std=c++20 -O3
	g++ $(PRE_PATH)/flatten.cpp -o $(EXE_PATH)/flatten -std=c++20 -O3

clean:
	rm __exe__/* -rf
	rmdir __exe__
