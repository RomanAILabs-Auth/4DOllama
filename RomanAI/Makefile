.PHONY: build run clean test test-final
build:
	r4 build romanai.r4d -o romanai
run: build
	./romanai
test-final:
	r4 run src/cli/main.r4d
clean:
	rm -f romanai
