import json
import logging
import time
from datetime import datetime

from gensim.models import KeyedVectors

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debugging.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_model():
    """Load pre-trained Word2Vec model (Google News)"""
    logger.info("Loading Word2Vec model...")
    start_time = time.time()

    model = KeyedVectors.load_word2vec_format(
        "GoogleNews-vectors-negative300.bin.gz",
        binary=True,
        limit=500000 # Loading all the words would take too much of my PC's resources...
    )

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds!")
    logger.info(f"Vocabulary size: {len(model.key_to_index)}")
    logger.info(f"Vector dimensions: {model.vector_size}")

    return model

def evaluate_analogies(model):
    """Evaluate word analogies"""
    logger.info("Evaluating analogies...")
    start_time=time.time()

    file_path = "questions-words.txt"
    analogies = model.evaluate_word_analogies(file_path)

    eval_time = time.time() - start_time
    eval_time_minutes = eval_time / 60
    logger.info(f"Evaluation completed in {eval_time_minutes:.2f} minutes!")

    return analogies

def analyze_results(analogies):
    """Analyze results and generate stats"""
    accuracy, sections = analogies

    logger.info("=" * 60)
    logger.info("DETAILED RESULTS BY CATEGORY")
    logger.info("=" * 60)

    total_correct = 0
    total_questions = 0

    for section in sections:
        section_name = section['section']
        correct = len(section['correct'])
        incorrect = len(section['incorrect'])
        total = correct + incorrect

        if total > 0:
            section_accuracy = correct / total
            logger.info(f"{section_name:30} {correct}/{total} ({section_accuracy:.2%})")

            total_correct += correct
            total_questions += total

    # The last section is about the total, so it is counted twice
    total_correct /= 2
    total_questions /= 2

    logger.info("=" * 60)
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    logger.info(f"{'OVERALL':30} {total_correct}/{total_questions} ({overall_accuracy:.2%})")

    return total_correct, total_questions, overall_accuracy

def save_results(accuracy, total_correct, total_questions, sections):
    """Save results to json file"""
    results = {
        "overall_accuracy": accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "categories": []
    }

    for section in sections:
        section_name = section['section']
        correct = len(section['correct'])
        incorrect = len(section['incorrect'])
        total = correct + incorrect

        results["categories"].append({
            "name": section_name,
            "correct": correct,
            "incorrect": incorrect,
            "total": total,
            "accuracy": correct / total if total > 0 else 0
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"results_{timestamp}.json"

    logger.info(f"Saving results to {json_filename}")
    logger.info("Results: " + str(results))

    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {json_filename}")

def show_examples(model):
    """Show some analogy examples"""
    logger.info("=" * 60)
    logger.info("EXAMPLE ANALOGIES")
    logger.info("=" * 60)

    test_analogies = [
        ("king", "man", "queen"),  # Expected: woman
        ("Paris", "France", "London"),  # Expected: England
        ("good", "better", "bad"),  # Expected: worse
        ("walk", "walked", "run"),  # Expected: ran
        ("cat", "cats", "dog"),  # Expected: dogs
    ]

    for i, (a, b, c) in enumerate(test_analogies, 1):
        try:
            result = model.most_similar(positive=[b, c], negative=[a], topn=1)

            predicted = result[0][0]
            confidence = result[0][1]

            logger.info(f"{i}. {a} : {b} :: {c} : {predicted} (confidence: {confidence:.3f})")
        except KeyError as e:
            logger.warning(f"{i}. {a} : {b} :: {c} : [Word not in vocabulary: {e}]")

def main():
    """Main function"""
    logger.info("Word Analogy Test with Word2Vec")
    logger.info("=" * 60)

    model = load_model()
    analogies = evaluate_analogies(model)
    total_correct, total_questions, accuracy = analyze_results(analogies)
    save_results(accuracy, total_correct, total_questions, analogies[1])
    show_examples(model)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {accuracy:.2%}")
    logger.info(f"Correct Answers: {total_correct:,}")
    logger.info(f"Total Questions: {total_questions:,}")

if __name__ == "__main__":
    main()