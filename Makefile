.DEFAULT_GOAL := help

help:

	@echo "Please use 'make <target>' where <target> is one of"
	@echo " setup		create a python environment"
	@echo " setup_conda	create and activate a conda environment, better for mac M1"
	@echo "	ingestion   1. data ingestion"
	@echo "	training    2. model training"
	@echo "	scoring     3. model scoring"
	@echo "	deploy      4. model deployment"
	@echo "	diagnos     5. model diagnostics"
	@echo "	report      6. model reporting"
	@echo "	run         7. run model REST API"
	@echo "	auto        8. check model drift and re-train model if needed"
	@echo "Check the Makefile to know exactly what each target is doing."


setup:
	python -m venv venv 
	source venv/bin/activate
setup_conda:
	conda env create --file environment.yml
install:
	pip install -r requirements.txt
ingestion:
	python ingestion.py
training:
	python training.py
scoring:
	python scoring.py
deploy:
	python deployment.py
diagnos:
	python diagnostics.py
report:
	python reporting.py
run:
	python app.py
auto:
	python fullprocess.py

		
