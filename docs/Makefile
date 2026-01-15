PANDOC = pandoc
# Use LuaLaTeX for proper Unicode/emoji support (✅ ❌)
# DejaVu Sans for main text, Noto Color Emoji for emoji characters
PANDOC_FLAGS = --toc --number-sections \
	--pdf-engine=lualatex \
	-V mainfont="DejaVu Sans" \
	-V monofont="DejaVu Sans Mono" \
	-H emoji-support.tex

# Pattern rule to convert any .md file to a .pdf in the output directory
output/%.pdf: %.md emoji-support.tex
	@mkdir -p output # Ensure output directory exists
	$(PANDOC) $(PANDOC_FLAGS) $< -o $@

# Optional: Default target to build all PDFs in the output directory
all: $(patsubst %.md,output/%.pdf,$(wildcard *.md))

# Optional: Clean up generated PDFs
clean:
	rm -f output/*.pdf
