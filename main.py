# CLI interface for the Local RAG Q&A Assistant using Microsoft Foundry Local
import sys
import os
import argparse
import atexit
import database
import rag_pipeline
import config

atexit.register(rag_pipeline.llm.unload_model)
atexit.register(rag_pipeline.embeddings.unload_model)

def parse_args():
    parser = argparse.ArgumentParser(description="Local RAG Assistant using Microsoft Foundry Local")
    parser.add_argument(
        "--reindex", "--ingest",
        action="store_true",
        help="Rebuild the knowledge base by chunking and embedding documents"
    )
    return parser.parse_args()

def show_banner(chunk_count):
    print("========================================")
    print("Local RAG Assistant")
    print("Microsoft Foundry Local")
    print("========================================")
    print(f"Knowledge base: {chunk_count} chunks")
    print("Type 'exit' or 'quit' to quit.")
    print("Type 'reindex' to rebuild the knowledge base.")
    print("----------------------------------------")

def main():
    args = parse_args()
    
    database.initialize_database()
    
    if args.reindex:
        print("Reindexing knowledge base as requested...")
        try:
            rag_pipeline.ingest_documents()
        except Exception as error:
            print(f"Reindex failed: {error}", file=sys.stderr)
            sys.exit(1)
        print("Reindexing complete. Exiting.")
        sys.exit(0)
        
    chunk_count = database.count_chunks()
    if chunk_count == 0:
        print("Knowledge base is empty.")
        # Check if documents directory exists and is not empty
        if not os.path.exists(config.DOCUMENTS_DIR) or not os.listdir(config.DOCUMENTS_DIR):
            print(f"Warning: The '{config.DOCUMENTS_DIR}' directory does not exist or has no documents.")
            print("Please add .txt, .md, or text-based .pdf files to the 'documents/' directory and run 'reindex'.")
        else:
            print("Documents found in local directory. Automatically building knowledge base...")
            try:
                _, chunk_count = rag_pipeline.ingest_documents()
            except Exception as e:
                print(f"Error during automatic ingestion: {e}")
                
    show_banner(chunk_count)
    
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not question:
            print("Assistant: Please enter a valid question.")
            continue
            
        # Check exit command
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        # Check reindex command
        if question.lower() == "reindex":
            print("\nRebuilding the knowledge base...")
            try:
                _, new_count = rag_pipeline.ingest_documents()
                print(f"Reindex successful. Knowledge base now has {new_count} chunks.\n")
            except Exception as e:
                print(f"Error during reindexing: {e}\n")
            continue
            
        # Process the question through the RAG pipeline
        print("\nAssistant is thinking...")
        result = rag_pipeline.query_pipeline(question)
        
        # Display the grounded answer
        print(f"\nAssistant:\n{result['answer']}")
        
        # Display sources if any were used
        if result["sources"]:
            print("\nSources:")
            for src in result["sources"]:
                print(f"* {src}")
        print("\n----------------------------------------")

if __name__ == "__main__":
    main()
