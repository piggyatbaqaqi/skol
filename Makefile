test: typecheck label_test file_test line_test paragraph_test finder_test mycobank_species_test iso4_test indexfungorum_authors_test

typecheck:
	mypy *.py

label_test:
	python3 label_test.py

file_test:
	python3 file_test.py

line_test:
	python3 label_test.py

paragraph_test:
	python3 label_test.py

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

