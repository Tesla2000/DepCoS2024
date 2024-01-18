# Makefile

# Variables
PYTHON = python3.11
VENV = venv
REQUIREMENTS_FILE = requirements.txt

# Targets
.PHONY: setup install clean

setup: $(VENV)/bin/activate
	@echo "Virtual environment is ready. Run 'source $(VENV)/bin/activate' to activate it."

$(VENV)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Activating virtual environment..."
	@source $(VENV)/bin/activate; \

install: $(VENV)/bin/activate
	@echo "Installing requirements..."
	@$(VENV)/bin/pip install -r $(REQUIREMENTS_FILE)
	@echo "Requirements installed successfully."

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@echo "Cleanup complete."
