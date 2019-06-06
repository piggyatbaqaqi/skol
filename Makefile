test: typecheck finder_test mycobank_species_test iso4_test indexfungorum_authors_test

typecheck:
	mypy *.py

finder_test:
	python3 finder_test.py

# mycobank_authors_test:
# 	python3 mycobank_authors_test.py

mycobank_species_test:
	python3 mycobank_species_test.py

iso4_test:
	python3 iso4_test.py

indexfungorum_authors_test:
	python3 indexfungorum_authors_test.py

