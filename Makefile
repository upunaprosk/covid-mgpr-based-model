PYTHON = python
CONDA = conda

.PHONY: install train_and_predict clean help

install:
	@echo "Installing required packages..."
	$(CONDA) create --name mgprcovid -f environment.yml
	$(CONDA) env update --prune -f environment.yml

train_and_predict:
	$(CONDA) run -n mgprcovid $(PYTHON) main.py

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__/
	rm -f *.pyc
	rm -f *.pyo
	rm -f *logs.txt
	rm -f *.png
