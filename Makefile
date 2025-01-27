objects := $(patsubst %.c,%.o,$(wildcard *.c))

OUT_DIR=out
BIN_DIR=bin

LLVM_HOME=/opt/homebrew/opt/llvm/bin
CC=$(LLVM_HOME)/clang
CFLAGS=-std=c11 -O2 -g3 -Wall -Wextra

DEPS = $(wildcard *.h)
OBJ = $(patsubst %.c,$(OUT_DIR)/%.o,$(wildcard *.c))


$(OUT_DIR): 
	mkdir -p "$(OUT_DIR)"

$(OUT_DIR)/%.o: %.c $(OUT_DIR) $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

${BIN_DIR}/%: $(OBJ)
	mkdir -p "$(BIN_DIR)"
	$(CC) $(CFLAGS) $(_APPS) $^ -o $@


.PHONY: clean
clean:
	rm -rf ${BIN_DIR}
	rm -rf ${OUT_DIR}