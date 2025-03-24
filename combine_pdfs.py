import os
import glob
from PyPDF2 import PdfMerger

def combine_pdfs():
    # Create a PDF merger object
    merger = PdfMerger()
    
    # Find all PDF files in current directory and subdirectories
    pdf_files = glob.glob("**/*.pdf", recursive=True)
    
    if not pdf_files:
        print("No PDF files found!")
        return
        
    print("Found the following PDFs:")
    for pdf in pdf_files:
        print(f"- {pdf}")
    
    # Add each PDF to the merger
    for pdf in pdf_files:
        try:
            merger.append(pdf)
        except Exception as e:
            print(f"Error adding {pdf}: {e}")
            continue
    
    # Write the combined PDF
    try:
        output_name = "combined_experiments.pdf"
        merger.write(output_name)
        merger.close()
        print(f"\nSuccessfully created {output_name}")
    except Exception as e:
        print(f"Error creating combined PDF: {e}")

if __name__ == "__main__":
    # Install PyPDF2 if not already installed
    try:
        import PyPDF2
    except ImportError:
        print("Installing PyPDF2...")
        import subprocess
        subprocess.run(["pip", "install", "PyPDF2"], check=True)
    
    combine_pdfs() 