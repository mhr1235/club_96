# Making It Source Notes

Source file: `/Users/markramos/Downloads/training corpora 2/mallory/Making It.pdf`

Identified source:

- Cindy Patton and Janis Kelly, *Making It: A Woman's Guide to Sex in the Age of AIDS*
- Firebrand Books, revised and updated 1988
- Illustrations by Alison Bechdel
- Spanish translation by Papusa Molina
- The cover also identifies it as *Haciendolo: Guia Sexual Para Mujeres en la Era del SIDA*

Use for Mallory:

- AIDS-era safer-sex media for women, lesbians, bisexual women, sex workers, and people who use IV drugs.
- Public health writing that combines practical instruction, comics, peer talk, risk assessment, pleasure, and survival politics.
- A model for Club 96 programming that makes health resources social, frank, erotic, bilingual, visual, and nonclinical.

Extraction notes:

- `pdftotext` does not extract usable text because the PDF is image/stencil-heavy.
- `pdftoppm` renders pages successfully.
- Rotating rendered pages with `sips --rotate 90` and then running `tesseract --psm 6` produces usable OCR for interior pages.
- Current chunks are paraphrased from OCR-confirmed sections and should be treated as historical public-health media, not current medical advice.
