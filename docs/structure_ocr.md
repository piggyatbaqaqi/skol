# Open-Source Layout-Aware OCR Pipeline Strategy

## 1. Project Profile & Constraints
* **Domain:** Taxonomic literature, books, and journal volumes.
* **Input Characteristics:** Mixed corpus consisting of born-digital PDFs alongside legacy scanned/image-only PDFs containing lower-quality, auto-generated OCR layers (e.g., historical Tesseract outputs).
* **Structural Anomalies:** Complex multi-column grids, inline figures positioned around page breaks, dense morphological matrices, species lists, and multi-page taxonomic tables. 
* **Hardware Architecture:** Local **Nvidia RTX 5090 (Laptop GPU)** with **24GB VRAM**.
* **Licensing & Access Model:** Open ecosystem released under a **GPL-3.0 license**. The destination platform is permanently free-access, with future operational sustainability driven by community crowdfunding (e.g., Patreon).

---

## 2. Recommended Processing Strategy
To balance computation efficiency with typographical fidelity (specifically preserving italics for *Genus species* names), the corpus should be routed through a two-tiered, adaptive Python pipeline.

Use code with caution.
￼
[ Input PDF ]
|
+------------+------------+
| |
[ Born-Digital ] [ Scanned/Legacy ]
| |
(Run Docling Pipeline) (Strip bad text layer)
| |
| (Run olmOCR Pipeline)
| |
+------------+------------+
|
[ Structured Markdown Output ]

### Track A: Born-Digital PDFs (Targeted Parsing)
* **Primary Engine:** **Docling (by IBM Research)**
* **License:** MIT (Permissive).
* **Execution Profile:** Low VRAM footprint (~4GB). Runs as a lightweight CPU/GPU hybrid engine.
* **Why it fits:** It extracts text natively without destroying crisp vector fonts. It possesses the most robust native cell-merging and bounding logic for dense table matrices spanning awkwardly across page boundaries. It isolates inline figures, saves them as individual images, and generates corresponding `![](image.png)` tags inline.

### Track B: Scanned/Legacy PDFs (Vision-Language OCR)
* **Primary Engine:** **olmOCR (by Allen Institute for AI)**
* **License:** Apache 2.0 (Permissive).
* **Execution Profile:** Medium VRAM footprint (~14–16GB), utilizing a highly optimized `Qwen2-VL-7B-Instruct` model core.
* **Why it fits:** Legacy OCR layers often corrupt italics, special characters, and scientific shorthand. This pipeline bypasses embedded OCR noise by rendering pages directly to images. The vision model views the entire graphical landscape at once, intuitively reconstructing reading orders across tight column shifts and cleanly spitting out Markdown tables and LaTeX math formatting.

---

## 3. Alternative Tool Evaluation

| Tool Framework | Licensing Profile | Strengths | Weaknesses / Considerations |
| :--- | :--- | :--- | :--- |
| **Docling** | MIT | Class-leading native table extraction and reconstruction. | Performance scales best when utilizing its native multi-threading features. |
| **olmOCR** | Apache 2.0 | Vision-first framework natively reading chaotic multi-column shifts. | Requires a modern VLM serving runtime (e.g., `vLLM` or `SGLang`). |
| **Marker** | GPL-3.0 | Exceptional at finding, isolating, and cleanly cropping visual plates/illustrations. | Can struggle with complex, dense scientific grid structures unless augmented with an LLM. |
| **MinerU** | AGPL-3.0 | High layout-retention capability and automated equation conversion. | Demands rigid environment configurations and heavy compute resource allocations. |

---

## 4. Hardware Optimization Strategies (RTX 5090 with 24GB VRAM)
* **vLLM Multi-Batching:** When routing legacy pages through the `olmOCR` vision engine, configure `vLLM` or `SGLang` to serve the 7B model. 24GB of VRAM allows enough headroom to process batches of **4 to 8 document pages concurrently**, maxing out CUDA compute throughput.
* **Pre-Filtering Classification:** Implement a preliminary step in Python (via a lightweight library like `pdfplumber` or `pypdf`) to inspect the document matrix. If the document contains native fonts, pass it to Docling. If it contains zero embedded text or an unindexed bitmap layer (Type 3 fonts), programmatically direct it to the image-rendering step for olmOCR.

---

## 5. Reference Implementation Framework

The following core pipeline logic uses `docling` to extract Markdown and physical assets simultaneously from structured documents:

```python
import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

def setup_converter(output_dir: Path) -> DocumentConverter:
    """Configures Docling to extract layouts, tables, and physical images."""
    pipeline_options = PdfPipelineOptions()
    
    # 1. Enable table structure extraction
    pipeline_options.do_table_structure = True
    
    # 2. Enable physical image/figure extraction to disk
    pipeline_options.images_scale = 2.0  # Extract high-res versions for biological plates
    pipeline_options.generate_pictures = True
    
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    return converter

def process_taxonomic_pdf(pdf_path: str, output_base_dir: str):
    pdf_path = Path(pdf_path)
    base_dir = Path(output_base_dir)
    
    # Isolate extraction output directories for this specific volume
    doc_output_dir = base_dir / pdf_path.stem
    images_output_dir = doc_output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing document architecture for: {pdf_path.name}...")
    converter = setup_converter(doc_output_dir)
    
    # Run parsing engine
    result = converter.convert(pdf_path)
    
    # Export and write physical cropped figures/plates to disk
    for element, _ in result.document.iterate_items():
        if element.label == "Picture":
            img_filename = f"figure_{element.id}.png"
            img_path = images_output_dir / img_filename
            with open(img_path, "wb") as f:
                element.image.pil_image.save(f, format="PNG")
    
    # Export finalized structured Markdown document
    markdown_content = result.document.export_to_markdown()
    md_path = doc_output_dir / f"{pdf_path.stem}.md"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Extraction successful. Assets deposited to: {doc_output_dir}")

if __name__ == "__main__":
    sample_pdf = "path/to/taxonomic_journal_volume.pdf"
    output_directory = "./taxonomy_corpus_output"
    process_taxonomic_pdf(sample_pdf, output_directory)
```
