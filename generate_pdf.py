#!/usr/bin/env python3
"""Convert patent application markdown to professional PDF."""
import markdown
from weasyprint import HTML

# Read markdown
with open('PATENT-APPLICATION.md', 'r') as f:
    md_content = f.read()

# Convert to HTML
html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Professional patent styling
html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: letter;
        margin: 1in 1.2in;
        @top-center {{
            content: "PROVISIONAL PATENT APPLICATION — AEZ-2026-001";
            font-size: 8pt;
            color: #666;
        }}
        @bottom-center {{
            content: "Page " counter(page) " of " counter(pages);
            font-size: 8pt;
            color: #666;
        }}
    }}
    body {{
        font-family: 'Times New Roman', 'DejaVu Serif', Georgia, serif;
        font-size: 12pt;
        line-height: 1.6;
        color: #1a1a1a;
    }}
    h1 {{
        font-size: 16pt;
        text-align: center;
        margin-top: 0.5in;
        margin-bottom: 0.3in;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }}
    h2 {{
        font-size: 14pt;
        margin-top: 0.4in;
        margin-bottom: 0.15in;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #222;
    }}
    h3 {{
        font-size: 12pt;
        font-weight: bold;
        margin-top: 0.25in;
        margin-bottom: 0.1in;
    }}
    h4 {{
        font-size: 11pt;
        font-weight: bold;
        margin-top: 0.2in;
        margin-bottom: 0.08in;
    }}
    p {{
        text-align: justify;
        margin-bottom: 0.15in;
    }}
    code {{
        font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
        font-size: 10pt;
        background: #f5f5f5;
        padding: 1px 4px;
        border-radius: 2px;
    }}
    pre {{
        font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
        font-size: 9.5pt;
        background: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 12px 16px;
        line-height: 1.4;
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }}
    pre code {{
        background: none;
        padding: 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 0.2in 0;
        font-size: 10pt;
    }}
    th, td {{
        border: 1px solid #999;
        padding: 6px 10px;
        text-align: left;
    }}
    th {{
        background: #f0f0f0;
        font-weight: bold;
    }}
    hr {{
        border: none;
        border-top: 1px solid #ccc;
        margin: 0.3in 0;
    }}
    strong {{
        font-weight: bold;
    }}
    ul, ol {{
        margin-left: 0.3in;
        margin-bottom: 0.15in;
    }}
    li {{
        margin-bottom: 0.05in;
    }}
    blockquote {{
        border-left: 3px solid #999;
        margin-left: 0.2in;
        padding-left: 0.15in;
        color: #444;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

# Generate PDF
output_path = 'PATENT-APPLICATION.pdf'
HTML(string=html_doc).write_pdf(output_path)
print(f"PDF generated: {output_path}")
