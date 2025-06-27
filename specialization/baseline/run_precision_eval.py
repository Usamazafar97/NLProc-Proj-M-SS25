from enhanced_pipeline import EnhancedRAGPipeline
import json

if __name__ == "__main__":
    # Initialize enhanced pipeline
    pipeline = EnhancedRAGPipeline()

    # Setup index with software reviews
    document_paths = ["retriever/software.json"]
    pipeline.setup_index(document_paths, force_rebuild=True)

    # Run test suite with Precision@k evaluation
    pipeline.run_test_suite(
        test_file="data/enhanced_test_inputs.json",
        answer_file="data/test_inputs.json",
        precision_k=5  # Change to 3 for Precision@3
    )

    # Print system statistics
    print(f"\n{'='*60}")
    print("📊 SYSTEM STATISTICS")
    print(f"{'='*60}")
    stats = pipeline.get_system_statistics()
    print(json.dumps(stats, indent=2)) 