.PHONY: all
all: wheel

.PHONY: wheel
wheel:
	python3 setup.py bdist_wheel sdist

.PHONY: clean
clean:
	rm -fr build dist *.egg-info

.PHONY: test
test:
	python -m unittest discover
