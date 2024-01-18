# Makefile

# Variables
PYTHON = python3.11
VENV = venv
REQUIREMENTS_FILE = requirements.txt

# Targets
.PHONY: setup clean

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Activating virtual environment..."
	@source venv/bin/activate

install: $(VENV)/bin/activate
	@echo "Installing requirements..."
	@$(VENV)/bin/pip install -r $(REQUIREMENTS_FILE)
	@echo "Requirements installed successfully."

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@echo "Cleanup complete."

