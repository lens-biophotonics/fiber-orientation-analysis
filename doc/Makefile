# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html: Makefile generate_rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp ../dist/foa3d-`python -c 'import foa3d; print(foa3d.__version__)'`-py3-none-any.whl build/html || true
	sed -i 's|doc/_static|_static|' build/html/index.html

generate_rst:
	sphinx-apidoc -faM --separate -o generated ../foa3d

clean:
	rm -fr _build generated
