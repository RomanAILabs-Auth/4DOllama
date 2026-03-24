.PHONY: all rust go test docker clean

all: rust go

rust:
	cd 4d-engine && cargo build --release

go: rust
	CGO_ENABLED=1 go build -o 4dollama ./cmd/4dollama

go-stub:
	CGO_ENABLED=0 go build -o 4dollama-stub ./cmd/4dollama

test:
	cd 4d-engine && cargo test
	go test ./...

docker:
	docker build -t fourdollama:latest .

clean:
	rm -f 4dollama 4dollama-stub
	cd 4d-engine && cargo clean
