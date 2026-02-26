IMAGES = images/skol_architecture.png images/skol_legend.png
DOCS = portfolio2.pdf

.PHONY: all clean
all: $(IMAGES) $(DOCS)

images/skol_architecture.png: images/skol_architecture.md
	python3 -c "\
	import mermaid as md; \
	from mermaid.graph import Graph; \
	content = open('$<').read(); \
	start = content.find('\`\`\`mermaid\n') + len('\`\`\`mermaid\n'); \
	end = content.find('\`\`\`', start); \
	md.Mermaid(Graph('skol-architecture', content[start:end])).to_png('$@')"

images/skol_legend.png: images/skol_legend.md
	python3 -c "\
	import mermaid as md; \
	from mermaid.graph import Graph; \
	content = open('$<').read(); \
	start = content.find('\`\`\`mermaid\n') + len('\`\`\`mermaid\n'); \
	end = content.find('\`\`\`', start); \
	md.Mermaid(Graph('skol-legend', content[start:end])).to_png('$@')"

portfolio2.pdf: portfolio2.md $(IMAGES)
	pandoc --toc --citeproc --number-sections --pdf-engine=xelatex -V "mainfont:DejaVu Sans" -V "monofont:DejaVu Sans Mono" -o portfolio2.pdf -f markdown -t pdf portfolio2.md

clean:
	rm -f $(IMAGES) $(DOCS)
