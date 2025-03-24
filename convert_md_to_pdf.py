import os
import glob
import subprocess
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
from bs4 import BeautifulSoup

def install_requirements():
    print("Installing required packages...")
    try:
        subprocess.run(["pip", "install", "markdown", "reportlab", "beautifulsoup4"], check=True)
    except Exception as e:
        print(f"Error installing packages: {e}")

def convert_md_to_pdf():
    try:
        import markdown
        from reportlab.platypus import SimpleDocTemplate
        from bs4 import BeautifulSoup
    except ImportError:
        print("Required packages are not installed. Installing now...")
        install_requirements()
        try:
            import markdown
            from reportlab.platypus import SimpleDocTemplate
            from bs4 import BeautifulSoup
        except ImportError:
            print("Failed to import packages even after installation. Please install manually.")
            return
    
    # Get all md files in current directory and subdirectories
    md_files = glob.glob("**/*.md", recursive=True)
    
    for md_file in md_files:
        try:
            # Determine output file path
            pdf_file = md_file.replace('.md', '.pdf')
            print(f"Converting {md_file} to {pdf_file}...")
            
            # Read markdown content
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert markdown to HTML
            html = markdown.markdown(md_content, extensions=['fenced_code', 'tables'])
            
            # Parse HTML using BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Create PDF document
            doc = SimpleDocTemplate(pdf_file, pagesize=letter)
            story = []
            
            # Define styles
            styles = getSampleStyleSheet()
            
            # Create a custom code style
            code_style = ParagraphStyle(
                name='CodeBlock',
                parent=styles['Normal'],
                fontName='Courier',
                fontSize=9,
                leading=12,
                backColor=colors.lightgrey,
                borderPadding=5,
                spaceAfter=10
            )
            
            # Define heading styles
            heading1_style = styles['Heading1']
            heading2_style = styles['Heading2']
            heading3_style = styles['Heading3']
            
            # Normal text style
            normal_style = styles['Normal']
            normal_style.leading = 14
            
            # Process all HTML elements
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre']):
                if element.name in ['h1', 'h2', 'h3', 'h4']:
                    style = {
                        'h1': heading1_style,
                        'h2': heading2_style,
                        'h3': heading3_style,
                        'h4': heading3_style
                    }.get(element.name, normal_style)
                    text = element.get_text()
                    story.append(Paragraph(text, style))
                    story.append(Spacer(1, 12))
                
                elif element.name == 'p':
                    text = element.get_text()
                    story.append(Paragraph(text, normal_style))
                    story.append(Spacer(1, 10))
                
                elif element.name == 'pre':
                    # Handle code blocks
                    code = element.get_text()
                    # Clean up the code to make it fit better
                    code = code.replace('\t', '    ')  # Replace tabs with spaces
                    story.append(Preformatted(code, code_style))
                    story.append(Spacer(1, 10))
            
            # Build the PDF document
            doc.build(story)
            print(f"Successfully converted {md_file}")
            
        except Exception as e:
            print(f"Error converting {md_file}: {e}")

if __name__ == "__main__":
    convert_md_to_pdf()
    print("Conversion complete!") 
    