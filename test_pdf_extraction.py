"""
Test script for financial PDF processing using pdfplumber
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from utils.document_processor import process_financial_pdf
import traceback

def test_apple_pdf():
    """Enhanced test with improved extraction quality analysis"""
    try:
        pdf_path = "data/uploads/2022 Q3 AAPL.pdf"
        print(f"🔍 Processing: {pdf_path}")
        print("=" * 60)
        
        # Process the document
        result = process_financial_pdf(pdf_path)
        
        # Summary statistics
        stats = result.extraction_stats
        print(f"📊 EXTRACTION SUMMARY:")
        print(f"   • Document: {result.metadata['file_name']}")
        print(f"   • Pages processed: {stats['total_pages']}")
        print(f"   • Tables extracted: {stats['tables_extracted']}")
        print(f"   • Text chunks: {stats['text_chunks']}")
        print(f"   • Table types found: {', '.join(stats['table_types'])}")
        
        # Data quality analysis
        print(f"\n🔬 DATA QUALITY ANALYSIS:")
        
        high_confidence_tables = [t for t in result.structured_tables if t.confidence_score > 0.7]
        medium_confidence_tables = [t for t in result.structured_tables if 0.4 <= t.confidence_score <= 0.7]
        low_confidence_tables = [t for t in result.structured_tables if t.confidence_score < 0.4]
        
        print(f"   • High confidence tables (>0.7): {len(high_confidence_tables)}")
        print(f"   • Medium confidence tables (0.4-0.7): {len(medium_confidence_tables)}")
        print(f"   • Low confidence tables (<0.4): {len(low_confidence_tables)}")
        
        success_rate = (len(high_confidence_tables) / len(result.structured_tables)) * 100 if result.structured_tables else 0
        print(f"   • Success rate: {success_rate:.1f}%")
        
        # Detailed table analysis (show first 5 high-confidence tables)
        print(f"\n📋 TOP EXTRACTED TABLES:")
        top_tables = sorted(result.structured_tables, key=lambda x: x.confidence_score, reverse=True)[:5]
        
        for i, table in enumerate(top_tables):
            print(f"\n   Table {i+1}:")
            print(f"      Page: {table.page}")
            print(f"      Type: {table.table_type}")
            print(f"      Dimensions: {len(table.data)} rows × {len(table.data.columns)} columns")
            print(f"      Confidence: {table.confidence_score:.2f}")
            print(f"      Headers: {table.headers[:3]}..." if len(table.headers) > 3 else f"      Headers: {table.headers}")
            
            # Data quality indicators
            if not table.data.empty:
                null_percentage = (table.data.isnull().sum().sum() / table.data.size) * 100
                print(f"      Missing data: {null_percentage:.1f}%")
                
                # Check for numeric data
                numeric_cols = table.data.select_dtypes(include=['number']).columns
                print(f"      Numeric columns: {len(numeric_cols)}")
                
                if len(numeric_cols) > 0:
                    print(f"      Sample numeric data:")
                    for col in numeric_cols[:2]:  # Show first 2 numeric columns
                        sample_vals = table.data[col].dropna().head(3).tolist()
                        print(f"        {col}: {sample_vals}")
        
        # Problem areas
        if low_confidence_tables:
            print(f"\n⚠️  LOW CONFIDENCE EXTRACTIONS TO REVIEW:")
            for table in low_confidence_tables[:3]:
                print(f"   • Page {table.page}, {table.table_type}, confidence: {table.confidence_score:.2f}")
        
        # Text extraction quality
        print(f"\n📝 TEXT EXTRACTION QUALITY:")
        if result.text_chunks:
            avg_word_count = sum(chunk['metadata']['word_count'] for chunk in result.text_chunks) / len(result.text_chunks)
            print(f"   • Total text chunks: {len(result.text_chunks)}")
            print(f"   • Average words per chunk: {avg_word_count:.0f}")
            print(f"   • Pages with text: {len(set(chunk['metadata']['page'] for chunk in result.text_chunks))}")
        
        # Sample text content
        if result.text_chunks:
            print(f"\n📄 SAMPLE TEXT CONTENT:")
            sample_chunk = result.text_chunks[len(result.text_chunks)//2]  # Middle chunk
            content = sample_chunk['content'][:300] + "..." if len(sample_chunk['content']) > 300 else sample_chunk['content']
            print(f"   Page {sample_chunk['metadata']['page']}: {content}")
        
        print(f"\n✅ EXTRACTION COMPLETE")
        print(f"   Overall Quality: {'Excellent' if success_rate > 80 else 'Good' if success_rate > 60 else 'Fair'}")
        print(f"   Ready for RAG processing: {'Yes' if len(high_confidence_tables) > 10 else 'Partial'}")
        
        return "SUCCESS: Enhanced PDF processing completed!"
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("\n🔧 Full traceback:")
        traceback.print_exc()
        return f"FAILED: {str(e)}"

if __name__ == "__main__":
    print("🚀 Enhanced Financial PDF Extraction Test")
    print("=" * 50)
    result = test_apple_pdf()
    print(f"\n🏁 Final Result: {result}")
