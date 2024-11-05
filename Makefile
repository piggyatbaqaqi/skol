test: typecheck file_test finder_test indexfungorum_authors_test iso4_test \
	label_test line_test mycobank_species_test paragraph_test taxon_test

typecheck:
	mypy *.py

%_test:
	python3 $@.py
