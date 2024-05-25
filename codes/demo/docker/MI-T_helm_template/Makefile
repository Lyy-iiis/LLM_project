podup:
	helm install mit --values project.yaml ./MI-T_chart/

connect:
	kubectl expose pod $(shell kubectl get pods | grep mi-t | awk '{ print $$1 }') --name=api-service --port=54224

poddown:
	helm delete mit
	kubectl delete service api-service