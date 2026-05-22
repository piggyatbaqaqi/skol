test: typecheck file_test finder_test indexfungorum_authors_test iso4_test \
	label_test line_test mycobank_species_test paragraph_test taxon_test

typecheck:
	mypy *.py

%_test:
	python3 $@.py

# Build the skol-gnservices .deb (gnfinder + gnparser as local
# HTTP services for the v4 span layer).  See packaging/skol-gnservices/
# and docs/v3_buildout.md §Phase F.
.PHONY: deb-gnservices
deb-gnservices:
	cd packaging/skol-gnservices && dpkg-buildpackage -us -uc -b
