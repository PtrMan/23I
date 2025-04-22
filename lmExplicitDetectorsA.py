# coding prompt  https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221GP4bQOwbnek5Jn7_CaaRrqw7KTuQmr6O%22%5D,%22action%22:%22open%22,%22userId%22:%22114048892711590756388%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# Define type aliases for clarity
TokenID = int
Context = Tuple[TokenID, ...]
ProbDistribution = Dict[TokenID, float]

@dataclass
class Detector:
    """Represents a detector that predicts a specific token given a context."""
    context: Context
    predicted_token: TokenID
    voting_strength: float = 0.0
    correct_predictions_epoch: int = 0
    incorrect_predictions_epoch: int = 0

    def reset_epoch_counters(self):
        """Resets the counters at the beginning of an epoch."""
        self.correct_predictions_epoch = 0
        self.incorrect_predictions_epoch = 0

    def calculate_accuracy(self) -> Optional[float]:
        """Calculates accuracy for the current epoch. Returns None if no predictions."""
        total_predictions = self.correct_predictions_epoch + self.incorrect_predictions_epoch
        if total_predictions == 0:
            return None # Or 0.0, depending on desired behavior for inactive detectors
        return self.correct_predictions_epoch / total_predictions

class DetectorLM(object):
    """
    A Language Model based on a collection of detectors.
    """
    def __init__(self,
                 n: int,
                 vocabulary_size: int,
                 min_accuracy_threshold: float = 0.1,
                 default_voting_strength: float = 1.0,
                 strength_update_factor: float = 5.0):
        """
        Initializes the Detector Language Model.

        Args:
            n: The context size (n-gram length) for detectors.
            vocabulary_size: The total number of unique tokens.
            min_accuracy_threshold: Minimum accuracy (correct / total) for a detector
                                     to survive pruning after an epoch.
            default_voting_strength: Initial strength assigned to new detectors.
            strength_update_factor: A factor used in updating voting strength based
                                     on accuracy (can be tuned).
        """
        if n < 1:
            raise ValueError("Context size n must be at least 1.")
        if vocabulary_size < 1:
            raise ValueError("Vocabulary size must be at least 1.")
        if not 0.0 <= min_accuracy_threshold <= 1.0:
             raise ValueError("min_accuracy_threshold must be between 0.0 and 1.0")

        self.n = n
        self.vocabulary_size = vocabulary_size
        self.min_accuracy_threshold = min_accuracy_threshold
        self.default_voting_strength = default_voting_strength
        self.strength_update_factor = strength_update_factor

        # Main storage: Maps a context tuple to a list of detectors for that context
        # Using defaultdict simplifies adding detectors for new contexts
        self.detectors: Dict[Context, List[Detector]] = defaultdict(list)

    def _get_context_ngram(self, sequence: List[TokenID]) -> Optional[Context]:
        """Extracts the relevant n-gram context from the end of a sequence."""
        if len(sequence) < self.n:
            return None # Not enough context
        return tuple(sequence[-self.n:])

    def _create_uniform_distribution(self) -> ProbDistribution:
        """Creates a uniform probability distribution over the vocabulary."""
        prob = 1.0 / self.vocabulary_size
        return {i: prob for i in range(self.vocabulary_size)}

    def predict(self, current_context_sequence: List[TokenID]) -> ProbDistribution:
        """
        Predicts the probability distribution for the next token.

        Args:
            current_context_sequence: A list of token IDs representing the current context.

        Returns:
            A dictionary mapping token IDs to their predicted probabilities.
        """
        context_ngram = self._get_context_ngram(current_context_sequence)

        if context_ngram is None:
            # Handle context shorter than n (e.g., return uniform)
            print(f"Warning: Context length {len(current_context_sequence)} < n ({self.n}). Returning uniform distribution.")
            return self._create_uniform_distribution()

        token_votes: Dict[TokenID, float] = defaultdict(float)
        total_strength_sum = 0.0

        # Check if any detectors match the context
        if context_ngram in self.detectors:
            active_detectors = self.detectors[context_ngram]
            for detector in active_detectors:
                predicted_token = detector.predicted_token
                strength = detector.voting_strength
                token_votes[predicted_token] += strength
                total_strength_sum += strength

        # Convert votes to a probability distribution
        if total_strength_sum > 0:
            probability_distribution: ProbDistribution = {
                token_id: vote_sum / total_strength_sum
                for token_id, vote_sum in token_votes.items()
            }
            # Ensure all tokens have a probability (even if 0) - optional but good practice
            for i in range(self.vocabulary_size):
                if i not in probability_distribution:
                    probability_distribution[i] = 0.0
            return probability_distribution
        else:
            # No detectors fired or all had zero strength. Fallback: Uniform distribution.
            return self._create_uniform_distribution()

    def _calculate_new_strength(self, accuracy: float, total_predictions: int) -> float:
        """
        Calculates the updated voting strength for a detector.
        Simple example: strength proportional to accuracy, scaled.
        Could be made more complex (e.g., factor in total_predictions for confidence).
        """
        # Ensure accuracy is not None before calculation
        if accuracy is None:
             return self.default_voting_strength # Or keep old strength? Needs policy. Let's default.

        # Example: Strength = base_factor * accuracy
        # We add a small epsilon to total_predictions to slightly boost confidence for more tested detectors
        # confidence_factor = math.log(total_predictions + 1.1) # Example using log
        # return accuracy * self.strength_update_factor * confidence_factor
        return accuracy * self.strength_update_factor


    def train_epoch(self, training_data: List[TokenID]):
        """
        Performs one epoch of training on the provided data.

        Args:
            training_data: A list of token IDs representing the training corpus.
        """
        print(f"Starting training epoch...")
        if len(training_data) <= self.n:
            print("Warning: Training data is too short for the given context size n.")
            return

        # 1. Reset epoch counters for all existing detectors
        print("Resetting detector counters...")
        detector_count = 0
        for detector_list in self.detectors.values():
            for detector in detector_list:
                detector.reset_epoch_counters()
                detector_count += 1
        print(f"Reset counters for {detector_count} detectors.")

        # 2. Iterate through the training data to update/create detectors
        print("Processing training data...")
        new_detectors_created = 0
        for i in range(len(training_data) - self.n):
            # Context is tokens from i to i+n-1
            context_ngram = tuple(training_data[i : i + self.n])
            # Target is the token at i+n
            actual_next_token = training_data[i + self.n]

            found_correct_predictor = False
            detectors_for_context = self.detectors[context_ngram] # Gets list or empty list

            # Update existing detectors for this context
            for detector in detectors_for_context:
                if detector.predicted_token == actual_next_token:
                    detector.correct_predictions_epoch += 1
                    found_correct_predictor = True
                else:
                    detector.incorrect_predictions_epoch += 1

            # Create a new detector if no existing detector correctly predicted
            # the actual next token for this specific context.
            if not found_correct_predictor:
                 # Check if a detector predicting this token *already exists* for this context
                 # (e.g., it might exist but wasn't deemed 'correct' if it was created in this epoch)
                 # This prevents creating exact duplicates within the same epoch pass.
                 already_exists = any(d.predicted_token == actual_next_token for d in detectors_for_context)

                 if not already_exists:
                     new_detector = Detector(
                         context=context_ngram,
                         predicted_token=actual_next_token,
                         voting_strength=self.default_voting_strength,
                         correct_predictions_epoch=1, # Made one correct prediction just now
                         incorrect_predictions_epoch=0
                     )
                     self.detectors[context_ngram].append(new_detector)
                     new_detectors_created += 1

            if (i + 1) % 10000 == 0: # Progress indicator
                 print(f"  Processed {i + 1}/{len(training_data) - self.n} contexts...")

        print(f"Finished processing data. Created {new_detectors_created} new detectors.")

        # --- End of Epoch Processing ---

        # 3. Prune weak detectors and update voting strengths
        print("Pruning weak detectors and updating strengths...")
        surviving_detectors_count = 0
        pruned_detectors_count = 0
        # Create a new dictionary to store survivors to avoid modifying while iterating
        new_detector_map: Dict[Context, List[Detector]] = defaultdict(list)

        for context_ngram, detector_list in self.detectors.items():
            surviving_detectors_for_context: List[Detector] = []
            for detector in detector_list:
                accuracy = detector.calculate_accuracy()
                total_predictions = detector.correct_predictions_epoch + detector.incorrect_predictions_epoch

                # Keep detector if accuracy meets threshold (and it was actually triggered)
                # Or keep if it wasn't triggered at all (accuracy is None)? Policy decision.
                # Let's keep untriggered ones for now, but don't update their strength.
                keep_detector = False
                if accuracy is not None:
                    if accuracy >= self.min_accuracy_threshold:
                        # Update strength based on performance
                        detector.voting_strength = self._calculate_new_strength(accuracy, total_predictions)
                        keep_detector = True
                    else:
                         pruned_detectors_count += 1 # Mark for pruning
                else:
                    # Detector was not triggered this epoch. Keep it with old strength?
                    # Or prune if inactive? Let's keep it for now.
                    keep_detector = True # Keep untriggered detector

                if keep_detector:
                    surviving_detectors_for_context.append(detector)
                    surviving_detectors_count += 1

            # If any detectors survived for this context, add them to the new map
            if surviving_detectors_for_context:
                new_detector_map[context_ngram] = surviving_detectors_for_context

        # Replace the old detector map with the pruned and updated one
        self.detectors = new_detector_map
        print(f"Pruning complete. Kept {surviving_detectors_count} detectors, pruned {pruned_detectors_count}.")
        print(f"Total contexts with detectors: {len(self.detectors)}")
        print("-" * 20)


# --- Example Usage ---

if __name__ == "__main__":
    # Simple vocabulary mapping (replace with actual tokenizer)
    vocab = {"<pad>": 0, "the": 1, "quick": 2, "brown": 3, "fox": 4, "jumps": 5, "over": 6, "lazy": 7, "dog": 8, ".": 9}
    inv_vocab = {v: k for k, v in vocab.items()}
    VOCAB_SIZE = len(vocab)

    # Sample training data (token IDs)
    # "the quick brown fox jumps over the lazy dog ."
    data1 = [1, 2, 3, 4, 5, 6, 1, 7, 8, 9]
    # "the lazy dog jumps ."
    data2 = [1, 7, 8, 5, 9]
    # "the quick fox jumps ."
    data3 = [1, 2, 4, 5, 9]
    # Repeat data to make patterns stronger
    training_data = (data1 * 5) + (data2 * 5) + (data3 * 5) + [1, 2, 3] # Add incomplete sequence at end

    # --- Model Initialization ---
    N_GRAM_SIZE = 2 # Use bigrams for context
    model = DetectorLM(
        n=N_GRAM_SIZE,
        vocabulary_size=VOCAB_SIZE,
        min_accuracy_threshold=0.3, # Keep detectors >30% accurate
        default_voting_strength=1.0,
        strength_update_factor=10.0 # Scale accuracy by 10 for strength
    )

    # --- Training ---
    NUM_EPOCHS = 3
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- EPOCH {epoch + 1} ---")
        # Shuffle data each epoch? Optional, can help generalization.
        # random.shuffle(training_data) # Be careful if state depends on order across epochs
        model.train_epoch(training_data)

    # --- Prediction ---
    print("\n--- PREDICTION ---")

    # Example 1: Context "the quick"
    context1 = [vocab["the"], vocab["quick"]]
    print(f"Context: {[inv_vocab[t] for t in context1]}")
    pred_dist1 = model.predict(context1)
    # Sort predictions by probability
    sorted_preds1 = sorted(pred_dist1.items(), key=lambda item: item[1], reverse=True)
    print("Predictions:")
    for token_id, prob in sorted_preds1[:5]: # Show top 5
        print(f"  Token: {inv_vocab[token_id]:<8} Probability: {prob:.4f}")
    # Expected: 'brown' or 'fox' should have high probability

    print("-" * 10)

    # Example 2: Context "lazy dog"
    context2 = [vocab["lazy"], vocab["dog"]]
    print(f"Context: {[inv_vocab[t] for t in context2]}")
    pred_dist2 = model.predict(context2)
    sorted_preds2 = sorted(pred_dist2.items(), key=lambda item: item[1], reverse=True)
    print("Predictions:")
    for token_id, prob in sorted_preds2[:5]:
        print(f"  Token: {inv_vocab[token_id]:<8} Probability: {prob:.4f}")
    # Expected: 'jumps' or '.' should have high probability

    print("-" * 10)

     # Example 3: Context "quick fox"
    context3 = [vocab["quick"], vocab["fox"]]
    print(f"Context: {[inv_vocab[t] for t in context3]}")
    pred_dist3 = model.predict(context3)
    sorted_preds3 = sorted(pred_dist3.items(), key=lambda item: item[1], reverse=True)
    print("Predictions:")
    for token_id, prob in sorted_preds3[:5]:
        print(f"  Token: {inv_vocab[token_id]:<8} Probability: {prob:.4f}")
    # Expected: 'jumps' should have high probability

    print("-" * 10)

    # Example 4: Unseen context (or less frequent)
    context4 = [vocab["jumps"], vocab["over"]]
    print(f"Context: {[inv_vocab[t] for t in context4]}")
    pred_dist4 = model.predict(context4)
    sorted_preds4 = sorted(pred_dist4.items(), key=lambda item: item[1], reverse=True)
    print("Predictions:")
    for token_id, prob in sorted_preds4[:5]:
        print(f"  Token: {inv_vocab[token_id]:<8} Probability: {prob:.4f}")
     # Expected: 'the' should have high probability

    # --- Inspect some detectors ---
    print("\n--- SAMPLE DETECTORS ---")
    context_to_inspect = (vocab["the"], vocab["quick"]) # tuple(context1)
    if context_to_inspect in model.detectors:
        print(f"Detectors for context {tuple(inv_vocab[t] for t in context_to_inspect)}:")
        for det in model.detectors[context_to_inspect]:
            acc = det.calculate_accuracy()
            acc_str = f"{acc:.2f}" if acc is not None else "N/A"
            print(f"  -> Predicts: {inv_vocab[det.predicted_token]:<8} Strength: {det.voting_strength:.2f} (Epoch Acc: {acc_str}, Correct: {det.correct_predictions_epoch}, Incorrect: {det.incorrect_predictions_epoch})")
    else:
        print(f"No detectors found for context {tuple(inv_vocab[t] for t in context_to_inspect)}")
