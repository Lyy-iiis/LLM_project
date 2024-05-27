podup:
	helm install llm --values project.yaml ./MI-T_chart/

connect:
	kubectl expose pod $(shell kubectl get pods | grep mi-t | awk '{ print $$1 }') --name=api-service --port=54224

poddown:
	helm delete llm
	kubectl delete service api-service
