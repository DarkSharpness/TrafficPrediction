EXE_PATH = __exe__

build:
	make init
	make compile
	make run

init:
	mkdir -p $(EXE_PATH)

run:
	./$(EXE_PATH)/train
	./$(EXE_PATH)/spliter
	./$(EXE_PATH)/pred
	./$(EXE_PATH)/cleaner

compile:
	g++ train.cpp -o $(EXE_PATH)/train -std=c++20 -O3
	g++ spliter.cpp -o $(EXE_PATH)/spliter -std=c++20 -O3
	g++ pred.cpp -o $(EXE_PATH)/pred -std=c++20 -O3
	g++ cleaner.cpp -o $(EXE_PATH)/cleaner -std=c++20 -O3

clean:
	rm __exe__/* -rf
	rm index/* -rf
	rmdir __exe__
	rmdir index
