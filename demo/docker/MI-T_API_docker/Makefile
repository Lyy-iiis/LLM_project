docker:
	docker build --platform linux/amd64 -t harbor.ai.iiis.co:9443/${USER}/api:latest .

push:
	docker push harbor.ai.iiis.co:9443/${USER}/api:latest

remove:
	docker rmi harbor.ai.iiis.co:9443/${USER}/api:latest