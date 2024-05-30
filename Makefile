EXE_PATH = __exe__
PRE_PATH = preprocess

all: init compile run

init:
	mkdir -p $(EXE_PATH)

run: compile
	./$(EXE_PATH)/train
	./$(EXE_PATH)/fixT
	./$(EXE_PATH)/residual
	./$(EXE_PATH)/predict
	./$(EXE_PATH)/fixP
	./$(EXE_PATH)/flatten
	./$(EXE_PATH)/spliter

targets = train fixT predict fixP flatten spliter result residual

compile: init $(foreach x,$(targets),$(EXE_PATH)/$(x))

$(EXE_PATH)/%: $(PRE_PATH)/%.cpp
	g++ $< -o $@ -std=c++20 -O3

clean:
	rm __exe__/* -rf
	rmdir __exe__
