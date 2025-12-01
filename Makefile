.PHONY: build up clean

build:
	docker-compose build

up:
	docker-compose up -d

clean:
	docker-compose down
