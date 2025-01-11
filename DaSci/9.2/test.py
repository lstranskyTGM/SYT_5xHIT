from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def main():
    # Define the directory where the fine-tuned model is saved
    save_directory = "bert-ner-conll2003"  # Update this if your save directory name is different

    # Load the fine-tuned model and tokenizer
    print("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModelForTokenClassification.from_pretrained(save_directory)

    # Create a pipeline for token classification
    print("Creating token classification pipeline...")
    ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Test the model with example sentences
    print("Testing the fine-tuned model...")
    test_sentences = [
        "Hugging Face is based in Brooklyn.",
        "Barack Obama was the 44th President of the United States.",
        "Apple Inc. is headquartered in Cupertino, California.",
        "The Eiffel Tower is located in Paris."
    ]

    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        predictions = ner_pipeline(sentence)
        print("Predictions:")
        for entity in predictions:
            print(
                f"Entity: {entity['word']}, "
                f"Type: {entity['entity_group']}, "
                f"Score: {entity['score']:.4f}, "
                f"Start: {entity['start']}, End: {entity['end']}"
            )


if __name__ == "__main__":
    main()
