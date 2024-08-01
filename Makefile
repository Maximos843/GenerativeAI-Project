build:
	docker build -t airi-project .

run:
	docker run --rm -it airi-project

dev: build
	docker run --rm -it airi-project