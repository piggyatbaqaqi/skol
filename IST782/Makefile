IMAGES = images/skol_architecture.png images/skol_legend.png images/skol_architecture_lr_left.png images/skol_architecture_lr_right.png images/skol_architecture_by_subproject_left.png images/skol_architecture_by_subproject_right.png
DOCS = portfolio2.pdf portfolio2.docx portfolio.pdf portfolio.docx

.PHONY: all clean
all: $(IMAGES) $(DOCS)

images/%.png: images/%.md
	python3 -c "\
	import mermaid as md; \
	from mermaid.graph import Graph; \
	content = open('$<').read(); \
	start = content.find('\`\`\`mermaid\n') + len('\`\`\`mermaid\n'); \
	end = content.find('\`\`\`', start); \
	md.Mermaid(Graph('$*', content[start:end])).to_png('$@')"

portfolio2.pdf: portfolio2.md $(IMAGES)
	pandoc --toc --citeproc --number-sections --pdf-engine=xelatex -V "mainfont:DejaVu Sans" -V "monofont:DejaVu Sans Mono" -o portfolio2.pdf -f markdown -t pdf portfolio2.md

portfolio.pdf: portfolio.md $(IMAGES)
	pandoc --toc --citeproc --number-sections --pdf-engine=xelatex -V "mainfont:DejaVu Sans" -V "monofont:DejaVu Sans Mono" -o portfolio.pdf -f markdown -t pdf portfolio.md

portfolio2.docx: portfolio2.md $(IMAGES)
	pandoc --toc --citeproc --number-sections -V "mainfont:DejaVu Sans" -V "monofont:DejaVu Sans Mono" -o portfolio2.docx -f markdown -t docx portfolio2.md

portfolio.docx: portfolio.md $(IMAGES)
	pandoc --toc --citeproc --number-sections -V "mainfont:DejaVu Sans" -V "monofont:DejaVu Sans Mono" -o portfolio.docx -f markdown -t docx portfolio.md

clean:
	rm -f $(IMAGES) $(DOCS)
