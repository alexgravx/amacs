setup:
	@pyenv virtualenv 3.10.12 optimization
install:
	@eval "$$(pyenv init -)" && \
	pyenv activate optimization && \
	pip install -r requirements.txt