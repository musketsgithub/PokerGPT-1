import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import json
from tqdm import tqdm
import re

token = 'hf_hRJppQUPLDjRqqNcJeDHaQaGvnltGZF'


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.ones(num_classes) if alpha is None else alpha

    def forward(self, inputs, targets):
        # Ensure inputs has correct shape [batch_size, num_classes]
        if len(inputs.shape) > 2:
            # If inputs has shape [batch_size, seq_len, num_classes]
            # Take the last token's prediction
            inputs = inputs[:, -1, :]

        # Get probabilities
        probs = F.softmax(inputs, dim=1)

        # One-hot encode targets if needed
        if len(targets.shape) == 1:
            # Check for out-of-bounds indices and fix them
            if targets.max() >= self.num_classes:
                targets = torch.clamp(targets, 0, self.num_classes - 1)
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            targets_one_hot = targets

        # Ensure alpha is on the right device
        alpha = self.alpha.to(inputs.device)

        # Get target probabilities    
        target_probs = (probs * targets_one_hot).sum(1)

        # Calculate focal weights
        focal_weight = (1 - target_probs) ** self.gamma

        # Apply class weights - handle index selection safely
        if len(targets.shape) == 1:
            # Ensure targets are in bounds
            safe_targets = torch.clamp(targets, 0, len(alpha) - 1)
            alpha_weight = alpha[safe_targets]
        else:
            alpha_weight = (alpha.unsqueeze(0) * targets_one_hot).sum(1)

        # Calculate loss with proper clamping
        focal_loss = -alpha_weight * focal_weight * torch.log(torch.clamp(target_probs, min=1e-8))

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HierarchicalPokerClassifier(nn.Module):
    def __init__(self, base_model, high_level_classes, bet_classes, class_weights_high=None, class_weights_bet=None):
        super().__init__()
        self.model = base_model
        self.hidden_size = 4096
        self.high_level_classes = high_level_classes  # Number of high-level classes (bet, check, call, fold)
        self.bet_classes = bet_classes  # Number of specific bet classes

        # First level classifier (high-level action)
        dropout_rate = 0.3
        self.dense1 = nn.Linear(self.hidden_size, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.classifier1 = nn.Linear(1024, high_level_classes)

        # Second level classifier (specific bet amount)
        self.dense2 = nn.Linear(self.hidden_size, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier2 = nn.Linear(1024, bet_classes)

        # Initialize weights
        nn.init.kaiming_normal_(self.dense1.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.xavier_uniform_(self.classifier1.weight)
        nn.init.zeros_(self.classifier1.bias)

        nn.init.kaiming_normal_(self.dense2.weight)
        nn.init.zeros_(self.dense2.bias)
        nn.init.xavier_uniform_(self.classifier2.weight)
        nn.init.zeros_(self.classifier2.bias)

        # Ensure float32 for stability
        self.dense1 = self.dense1.float()
        self.ln1 = self.ln1.float()
        self.classifier1 = self.classifier1.float()

        self.dense2 = self.dense2.float()
        self.ln2 = self.ln2.float()
        self.classifier2 = self.classifier2.float()

        # Loss functions with focal loss for both classifiers
        if class_weights_high is not None:
            self.loss_fn1 = FocalLoss(high_level_classes, alpha=class_weights_high, gamma=2.0)
        else:
            self.loss_fn1 = FocalLoss(high_level_classes, gamma=2.0)

        if class_weights_bet is not None:
            self.loss_fn2 = FocalLoss(bet_classes, alpha=class_weights_bet, gamma=2.0)
        else:
            self.loss_fn2 = FocalLoss(bet_classes, gamma=2.0)

    def extract_features(self, input_ids, attention_mask):
        """Extract features from base model"""
        with torch.no_grad():  # Don't need gradients for the base model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Get only the last hidden state to save memory
            if hasattr(outputs, 'hidden_states'):
                last_hidden_state = outputs.hidden_states[-1]
            else:
                # Fallback if the model doesn't return hidden_states directly
                last_hidden_state = outputs.last_hidden_state

            # Get the last token representation for each sequence
            # This avoids dimensional issues with variable sequence lengths
            last_token_pos = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]

            # Handle empty sequences
            last_token_pos = torch.clamp(last_token_pos, min=0)

            # Extract the last token's hidden state for each sequence
            pooled_output = torch.stack([
                last_hidden_state[i, pos, :]
                for i, pos in enumerate(last_token_pos)
            ])

        return pooled_output.float()  # Ensure float32

    def forward(self, input_ids, attention_mask, high_level_labels=None, bet_labels=None, is_bet_sample=None):
        pooled_output = self.extract_features(input_ids, attention_mask)

        # First classifier (high-level action)
        x1 = self.dense1(pooled_output)
        x1 = self.ln1(x1)
        x1 = F.gelu(x1)
        x1 = self.dropout1(x1)
        high_level_logits = self.classifier1(x1)

        # Second classifier (specific bet amount)
        x2 = self.dense2(pooled_output)
        x2 = self.ln2(x2)
        x2 = F.gelu(x2)
        x2 = self.dropout2(x2)
        bet_logits = self.classifier2(x2)

        if high_level_labels is not None:
            # For training mode
            # Handle one-hot labels if needed
            if len(high_level_labels.shape) > 1:
                high_level_indices = torch.argmax(high_level_labels, dim=1)
            else:
                high_level_indices = high_level_labels

            # Calculate loss for first classifier
            loss1 = self.loss_fn1(high_level_logits, high_level_indices)

            total_loss = loss1

            # Only calculate bet loss for bet samples
            if bet_labels is not None and is_bet_sample is not None:
                # Filter bet samples
                bet_samples = is_bet_sample.bool()
                if bet_samples.sum() > 0:
                    # Handle one-hot labels for bet classifier if needed
                    if len(bet_labels.shape) > 1:
                        bet_indices = torch.argmax(bet_labels[bet_samples], dim=1)
                    else:
                        bet_indices = bet_labels[bet_samples]

                    # Calculate loss for second classifier (only on bet samples)
                    loss2 = self.loss_fn2(bet_logits[bet_samples], bet_indices)

                    # Weighted sum of losses (can adjust weights if needed)
                    total_loss = loss1 + loss2

            # Debug accuracy for high-level classifier
            with torch.no_grad():
                high_preds = torch.argmax(high_level_logits, dim=1)
                high_correct = (high_preds == high_level_indices).sum().item()
                high_total = high_level_indices.size(0)
                high_acc = high_correct / high_total if high_total > 0 else 0

                # Calculate bet accuracy if there are bet samples
                if bet_labels is not None and is_bet_sample is not None and bet_samples.sum() > 0:
                    bet_preds = torch.argmax(bet_logits[bet_samples], dim=1)
                    if len(bet_labels.shape) > 1:
                        bet_targets = torch.argmax(bet_labels[bet_samples], dim=1)
                    else:
                        bet_targets = bet_labels[bet_samples]
                    bet_correct = (bet_preds == bet_targets).sum().item()
                    bet_total = bet_targets.size(0)
                    bet_acc = bet_correct / bet_total if bet_total > 0 else 0
                    print(f"Batch accuracies: High-level={high_acc:.4f}, Bet={bet_acc:.4f}")
                else:
                    print(f"Batch accuracy: High-level={high_acc:.4f}")

            return {
                "loss": total_loss,
                "high_level_logits": high_level_logits,
                "bet_logits": bet_logits
            }

        return {
            "high_level_logits": high_level_logits,
            "bet_logits": bet_logits
        }


class HierarchicalPokerDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_length = max_length

        # Create high-level label mapping (bets, checks, calls, folds)
        self.high_level_mapping = {
            "checks": 0,
            "calls": 1,
            "folds": 2,
            "bets": 3  # All numeric labels will be mapped to "bets"
        }

        # Process labels to get high-level and bet-specific labels
        self.high_level_labels = []
        self.bet_labels = []
        self.is_bet_sample = []
        self.bet_value_to_id = {}

        # First pass: identify all unique bet values
        bet_values = []
        for label in labels:
            # Check if label is numeric (a bet amount)
            if re.match(r'^\d+(\.\d+)?$', str(label)):
                bet_values.append(label)

        # Create bet value mapping
        unique_bet_values = sorted(set(bet_values))
        self.bet_value_to_id = {value: i for i, value in enumerate(unique_bet_values)}
        self.id_to_bet_value = {i: value for i, value in enumerate(unique_bet_values)}

        # Second pass: assign high-level and bet-specific labels
        for label in labels:
            if re.match(r'^\d+(\.\d+)?$', str(label)):
                # It's a bet
                self.high_level_labels.append(self.high_level_mapping["bets"])
                self.bet_labels.append(self.bet_value_to_id[label])
                self.is_bet_sample.append(1)
            elif label in self.high_level_mapping:
                # It's checks, calls, or folds
                self.high_level_labels.append(self.high_level_mapping[label])
                self.bet_labels.append(0)  # Placeholder
                self.is_bet_sample.append(0)
            else:
                # Unrecognized label, default to checks
                default_action = "checks" if "checks" in self.high_level_mapping else "check"
                print(f"Warning: Unrecognized label '{label}'. Defaulting to '{default_action}'.")
                self.high_level_labels.append(self.high_level_mapping[default_action])
                self.bet_labels.append(0)  # Placeholder
                self.is_bet_sample.append(0)

        self.num_high_level_classes = len(self.high_level_mapping)
        self.num_bet_classes = len(self.bet_value_to_id)

        print(f"High-level classes: {self.num_high_level_classes}")
        print(f"Bet-specific classes: {self.num_bet_classes}")

        # Calculate class distribution
        high_level_counter = Counter(self.high_level_labels)
        bet_counter = Counter([self.bet_labels[i] for i in range(len(self.bet_labels)) if self.is_bet_sample[i]])

        self.high_level_counts = high_level_counter
        self.bet_counts = bet_counter

        print(f"High-level distribution: {dict(high_level_counter)}")
        print(f"Bet distribution: Top 10 amounts: {bet_counter.most_common(10)}")

        # Tokenize prompts
        print("Tokenizing prompts...")
        self.encodings = tokenizer(
            prompts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        # Create tensors
        self.high_level_label_tensor = torch.tensor(self.high_level_labels, dtype=torch.long)
        self.bet_label_tensor = torch.tensor(self.bet_labels, dtype=torch.long)
        self.is_bet_sample_tensor = torch.tensor(self.is_bet_sample, dtype=torch.long)

        # Calculate class weights
        high_level_weights = []
        for i in range(self.num_high_level_classes):
            count = high_level_counter.get(i, 0)
            weight = len(self.high_level_labels) / (count * self.num_high_level_classes) if count > 0 else 1.0
            high_level_weights.append(weight)

        bet_weights = []
        total_bet_samples = sum(self.is_bet_sample)
        for i in range(self.num_bet_classes):
            count = bet_counter.get(i, 0)
            weight = total_bet_samples / (count * self.num_bet_classes) if count > 0 else 1.0
            bet_weights.append(weight)

        self.high_level_weights = torch.tensor(high_level_weights, dtype=torch.float)
        self.bet_weights = torch.tensor(bet_weights, dtype=torch.float)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'high_level_labels': self.high_level_label_tensor[idx],
            'bet_labels': self.bet_label_tensor[idx],
            'is_bet_sample': self.is_bet_sample_tensor[idx]
        }

    def __len__(self):
        return len(self.high_level_label_tensor)

    def get_num_high_level_classes(self):
        return self.num_high_level_classes

    def get_num_bet_classes(self):
        return self.num_bet_classes

    def get_high_level_weights(self):
        return self.high_level_weights

    def get_bet_weights(self):
        return self.bet_weights

    def get_high_level_mapping(self):
        return self.high_level_mapping

    def get_bet_value_mapping(self):
        return self.bet_value_to_id


def compute_hierarchical_metrics(pred, is_bet_sample):
    """Calculate accuracy and per-class metrics for hierarchical model"""
    high_level_logits, bet_logits = pred.predictions

    # Convert logits to predictions
    high_level_preds = np.argmax(high_level_logits, axis=1)
    bet_preds = np.argmax(bet_logits, axis=1)

    # Get labels
    if isinstance(pred.label_ids[0], torch.Tensor):
        high_level_labels = pred.label_ids[0].cpu().numpy()
        bet_labels = pred.label_ids[1].cpu().numpy()
    else:
        high_level_labels, bet_labels = pred.label_ids

    # Handle one-hot encoded labels
    if len(high_level_labels.shape) > 1 and high_level_labels.shape[1] > 1:
        high_level_labels = np.argmax(high_level_labels, axis=1)
    if len(bet_labels.shape) > 1 and bet_labels.shape[1] > 1:
        bet_labels = np.argmax(bet_labels, axis=1)

    # Calculate high-level accuracy
    high_level_accuracy = np.mean(high_level_preds == high_level_labels)

    # Convert is_bet_sample to boolean mask if needed
    if isinstance(is_bet_sample, torch.Tensor):
        is_bet_sample = is_bet_sample.cpu().numpy().astype(bool)

    # Calculate bet accuracy only for bet samples
    bet_samples_mask = is_bet_sample.astype(bool)
    if np.any(bet_samples_mask):
        bet_accuracy = np.mean(bet_preds[bet_samples_mask] == bet_labels[bet_samples_mask])
    else:
        bet_accuracy = 0.0

    # Calculate combined accuracy (correct on both levels for bet samples, correct on high level for non-bet samples)
    combined_correct = 0
    for i in range(len(high_level_preds)):
        if bet_samples_mask[i]:
            if high_level_preds[i] == high_level_labels[i] and bet_preds[i] == bet_labels[i]:
                combined_correct += 1
        else:
            if high_level_preds[i] == high_level_labels[i]:
                combined_correct += 1

    combined_accuracy = combined_correct / len(high_level_preds)

    metrics = {
        "high_level_accuracy": high_level_accuracy,
        "bet_accuracy": bet_accuracy,
        "combined_accuracy": combined_accuracy
    }

    print(f"High-level accuracy: {high_level_accuracy:.4f}")
    print(f"Bet accuracy: {bet_accuracy:.4f}")
    print(f"Combined accuracy: {combined_accuracy:.4f}")

    return metrics


class HierarchicalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract is_bet_sample from kwargs
        self.is_bet_sample = kwargs.pop("is_bet_sample", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        high_level_labels = inputs.get("high_level_labels")
        bet_labels = inputs.get("bet_labels")
        is_bet_sample = inputs.get("is_bet_sample")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            high_level_labels=high_level_labels,
            bet_labels=bet_labels,
            is_bet_sample=is_bet_sample
        )

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            # Get logits
            high_level_logits = outputs["high_level_logits"]
            bet_logits = outputs["bet_logits"]

            # Get labels
            high_level_labels = inputs.get("high_level_labels")
            bet_labels = inputs.get("bet_labels")

            # Return in expected format for compute_metrics
            return (None, [high_level_logits, bet_logits], [high_level_labels, bet_labels])


def train_hierarchical_poker_model(data_path):
    # Set device - if memory is an issue, force CPU
    if torch.cuda.is_available():
        # Check GPU memory
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if free_memory < 8 * 1024 * 1024 * 1024:  # Less than 8GB free
            print("Low GPU memory, using CPU instead")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load dataset
    try:
        with open(data_path, 'r') as json_file:
            dataset = json.load(json_file)
    except FileNotFoundError:
        print(f"Dataset file not found: {data_path}")
        return

    # Extract data
    prompts = dataset['Prompts']
    labels = dataset['Correct Moves']

    # Limit dataset size if needed to prevent OOM
    max_samples = 10000  # Adjust based on memory constraints
    if len(prompts) > max_samples:
        print(f"Limiting dataset to {max_samples} samples to avoid OOM")
        indices = np.random.choice(len(prompts), max_samples, replace=False)
        prompts = [prompts[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Calculate class distribution
    label_counts = Counter(labels)
    print(f"Original total classes: {len(label_counts)}")
    print(f"Top original classes: {label_counts.most_common(10)}")

    # Split data (non-stratified to avoid errors with too many classes)
    train_prompts, temp_prompts, train_labels, temp_labels = train_test_split(
        prompts, labels, test_size=0.1, random_state=42
    )
    val_prompts, test_prompts, val_labels, test_labels = train_test_split(
        temp_prompts, temp_labels, test_size=0.5, random_state=42
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # Use a smaller sequence length to save memory
    max_length = 256

    # Create hierarchical datasets
    print("Creating hierarchical datasets...")
    train_dataset = HierarchicalPokerDataset(train_prompts, train_labels, tokenizer, max_length=max_length)
    val_dataset = HierarchicalPokerDataset(val_prompts, val_labels, tokenizer, max_length=max_length)
    test_dataset = HierarchicalPokerDataset(test_prompts, test_labels, tokenizer, max_length=max_length)

    # Get number of classes and class weights
    num_high_level_classes = train_dataset.get_num_high_level_classes()
    num_bet_classes = train_dataset.get_num_bet_classes()
    high_level_weights = train_dataset.get_high_level_weights()
    bet_weights = train_dataset.get_bet_weights()

    print(f"Number of high-level classes: {num_high_level_classes}")
    print(f"Number of bet-specific classes: {num_bet_classes}")

    # Load base model with minimal config to save memory
    print("Loading base model with memory optimizations...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "musketshugging/SFT225",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=token
    )

    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Create hierarchical classifier
    print("Initializing hierarchical classifier...")
    classifier = HierarchicalPokerClassifier(
        base_model,
        num_high_level_classes,
        num_bet_classes,
        high_level_weights.to(device),
        bet_weights.to(device)
    ).to(device)

    # Model has been successfully initialized
    model_loaded = True

    # Use smaller batch sizes and more accumulation steps to save memory
    training_args = TrainingArguments(
        output_dir="./hierarchical_poker_classifier",
        num_train_epochs=1,  # Reduced epochs for faster training
        per_device_train_batch_size=1,  # Smaller batch size
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,  # Less frequent evaluation
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="combined_accuracy",
        report_to="none",
        gradient_accumulation_steps=16,  # More accumulation steps
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,  # Avoid extra memory usage from workers
        remove_unused_columns=False,
        optim="adamw_torch",  # Use standard optimizer
        max_grad_norm=1.0,  # Clip gradients to prevent instability
    )

    # Get is_bet_sample for validation dataset
    val_is_bet_sample = val_dataset.is_bet_sample_tensor

    # Create custom trainer
    def compute_metrics_wrapper(pred):
        return compute_hierarchical_metrics(pred, val_is_bet_sample)

    trainer = HierarchicalTrainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_wrapper,
        is_bet_sample=val_is_bet_sample
    )

    try:
        # Train model
        print("Starting training...")
        trainer.train()

        # Evaluate on test set
        print("Evaluating on test set...")
        test_is_bet_sample = test_dataset.is_bet_sample_tensor

        def compute_test_metrics(pred):
            return compute_hierarchical_metrics(pred, test_is_bet_sample)

        trainer.compute_metrics = compute_test_metrics
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")

        # Save model
        print("Saving model...")
        trainer.save_model("./final_hierarchical_poker_classifier")

        # Save mappings
        high_level_mapping = train_dataset.get_high_level_mapping()
        bet_value_mapping = train_dataset.get_bet_value_mapping()

        mappings = {
            "high_level_mapping": high_level_mapping,
            "bet_value_mapping": bet_value_mapping
        }

        with open('./final_hierarchical_poker_classifier/label_mappings.json', 'w') as f:
            json.dump(mappings, f)

        print("Training complete!")
        return classifier, mappings

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("CUDA out of memory error. Try running with CPU or reducing dataset size.")
        else:
            print(f"Runtime error during training: {e}")

        # Clean up to free memory
        if 'classifier' in locals():
            del classifier
        if 'base_model' in locals():
            del base_model
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache()
        return None, None


if __name__ == "__main__":
    train_hierarchical_poker_model('pros_data1.json', str(label))
    # It's a bet
    self.high_level_labels.append(self.high_level_mapping["bets"])
    self.bet_labels.append(self.bet_value_to_id[label])
    self.is_bet_sample.append(1)
    elif label in self.high_level_mapping:
    # It's checks, calls, or folds
    self.high_level_labels.append(self.high_level_mapping[label])
    self.bet_labels.append(0)  # Placeholder
    self.is_bet_sample.append(0)
    else:
    # Unrecognized label, default to check
    print(f"Warning: Unrecognized label '{label}'. Defaulting to 'check'.")
    self.high_level_labels.append(self.high_level_mapping["check"])
    self.bet_labels.append(0)  # Placeholder
    self.is_bet_sample.append(0)

    self.num_high_level_classes = len(self.high_level_mapping)
    self.num_bet_classes = len(self.bet_value_to_id)

    print(f"High-level classes: {self.num_high_level_classes}")
    print(f"Bet-specific classes: {self.num_bet_classes}")

    # Calculate class distribution
    high_level_counter = Counter(self.high_level_labels)
    bet_counter = Counter([self.bet_labels[i] for i in range(len(self.bet_labels)) if self.is_bet_sample[i]])

    self.high_level_counts = high_level_counter
    self.bet_counts = bet_counter

    print(f"High-level distribution: {dict(high_level_counter)}")
    print(f"Bet distribution: Top 10 amounts: {bet_counter.most_common(10)}")

    # Tokenize prompts
    print("Tokenizing prompts...")
    self.encodings = tokenizer(
        prompts,
        truncation = True,
        padding = 'max_length',
        max_length = max_length,
        return_tensors = 'pt'
    )

# Create tensors
self.high_level_label_tensor = torch.tensor(self.high_level_labels, dtype=torch.long)
self.bet_label_tensor = torch.tensor(self.bet_labels, dtype=torch.long)
self.is_bet_sample_tensor = torch.tensor(self.is_bet_sample, dtype=torch.long)

# Calculate class weights
high_level_weights = []
for i in range(self.num_high_level_classes):
    count = high_level_counter.get(i, 0)
weight = len(self.high_level_labels) / (count * self.num_high_level_classes) if count > 0 else 1.0
high_level_weights.append(weight)

bet_weights = []
total_bet_samples = sum(self.is_bet_sample)
for i in range(self.num_bet_classes):
    count = bet_counter.get(i, 0)
weight = total_bet_samples / (count * self.num_bet_classes) if count > 0 else 1.0
bet_weights.append(weight)

self.high_level_weights = torch.tensor(high_level_weights, dtype=torch.float)
self.bet_weights = torch.tensor(bet_weights, dtype=torch.float)


def __getitem__(self, idx):
    return {
        'input_ids': self.encodings['input_ids'][idx],
        'attention_mask': self.encodings['attention_mask'][idx],
        'high_level_labels': self.high_level_label_tensor[idx],
        'bet_labels': self.bet_label_tensor[idx],
        'is_bet_sample': self.is_bet_sample_tensor[idx]
    }


def __len__(self):
    return len(self.high_level_label_tensor)


def get_num_high_level_classes(self):
    return self.num_high_level_classes


def get_num_bet_classes(self):
    return self.num_bet_classes


def get_high_level_weights(self):
    return self.high_level_weights


def get_bet_weights(self):
    return self.bet_weights


def get_high_level_mapping(self):
    return self.high_level_mapping


def get_bet_value_mapping(self):
    return self.bet_value_to_id


def compute_hierarchical_metrics(pred, is_bet_sample):
    """Calculate accuracy and per-class metrics for hierarchical model"""
    high_level_logits, bet_logits = pred.predictions

    # Convert logits to predictions
    high_level_preds = np.argmax(high_level_logits, axis=1)
    bet_preds = np.argmax(bet_logits, axis=1)

    # Get labels
    if isinstance(pred.label_ids[0], torch.Tensor):
        high_level_labels = pred.label_ids[0].cpu().numpy()
        bet_labels = pred.label_ids[1].cpu().numpy()
    else:
        high_level_labels, bet_labels = pred.label_ids

    # Handle one-hot encoded labels
    if len(high_level_labels.shape) > 1 and high_level_labels.shape[1] > 1:
        high_level_labels = np.argmax(high_level_labels, axis=1)
    if len(bet_labels.shape) > 1 and bet_labels.shape[1] > 1:
        bet_labels = np.argmax(bet_labels, axis=1)

    # Calculate high-level accuracy
    high_level_accuracy = np.mean(high_level_preds == high_level_labels)

    # Convert is_bet_sample to boolean mask if needed
    if isinstance(is_bet_sample, torch.Tensor):
        is_bet_sample = is_bet_sample.cpu().numpy().astype(bool)

    # Calculate bet accuracy only for bet samples
    bet_samples_mask = is_bet_sample.astype(bool)
    if np.any(bet_samples_mask):
        bet_accuracy = np.mean(bet_preds[bet_samples_mask] == bet_labels[bet_samples_mask])
    else:
        bet_accuracy = 0.0

    # Calculate combined accuracy (correct on both levels for bet samples, correct on high level for non-bet samples)
    combined_correct = 0
    for i in range(len(high_level_preds)):
        if bet_samples_mask[i]:
            if high_level_preds[i] == high_level_labels[i] and bet_preds[i] == bet_labels[i]:
                combined_correct += 1
        else:
            if high_level_preds[i] == high_level_labels[i]:
                combined_correct += 1

    combined_accuracy = combined_correct / len(high_level_preds)

    metrics = {
        "high_level_accuracy": high_level_accuracy,
        "bet_accuracy": bet_accuracy,
        "combined_accuracy": combined_accuracy
    }

    print(f"High-level accuracy: {high_level_accuracy:.4f}")
    print(f"Bet accuracy: {bet_accuracy:.4f}")
    print(f"Combined accuracy: {combined_accuracy:.4f}")

    return metrics


class HierarchicalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract is_bet_sample from kwargs
        self.is_bet_sample = kwargs.pop("is_bet_sample", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        high_level_labels = inputs.get("high_level_labels")
        bet_labels = inputs.get("bet_labels")
        is_bet_sample = inputs.get("is_bet_sample")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            high_level_labels=high_level_labels,
            bet_labels=bet_labels,
            is_bet_sample=is_bet_sample
        )

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            # Get logits
            high_level_logits = outputs["high_level_logits"]
            bet_logits = outputs["bet_logits"]

            # Get labels
            high_level_labels = inputs.get("high_level_labels")
            bet_labels = inputs.get("bet_labels")

            # Return in expected format for compute_metrics
            return (None, [high_level_logits, bet_logits], [high_level_labels, bet_labels])


def train_hierarchical_poker_model(data_path):
    # Set device - if memory is an issue, force CPU
    if torch.cuda.is_available():
        # Check GPU memory
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if free_memory < 8 * 1024 * 1024 * 1024:  # Less than 8GB free
            print("Low GPU memory, using CPU instead")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load dataset
    try:
        with open(data_path, 'r') as json_file:
            dataset = json.load(json_file)
    except FileNotFoundError:
        print(f"Dataset file not found: {data_path}")
        return

    # Extract data
    prompts = dataset['Prompts']
    labels = dataset['Correct Moves']

    # Limit dataset size if needed to prevent OOM
    max_samples = 10000  # Adjust based on memory constraints
    if len(prompts) > max_samples:
        print(f"Limiting dataset to {max_samples} samples to avoid OOM")
        indices = np.random.choice(len(prompts), max_samples, replace=False)
        prompts = [prompts[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Calculate class distribution
    label_counts = Counter(labels)
    print(f"Original total classes: {len(label_counts)}")
    print(f"Top original classes: {label_counts.most_common(10)}")

    # Split data (non-stratified to avoid errors with too many classes)
    train_prompts, temp_prompts, train_labels, temp_labels = train_test_split(
        prompts, labels, test_size=0.1, random_state=42
    )
    val_prompts, test_prompts, val_labels, test_labels = train_test_split(
        temp_prompts, temp_labels, test_size=0.5, random_state=42
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # Use a smaller sequence length to save memory
    max_length = 256

    # Create hierarchical datasets
    print("Creating hierarchical datasets...")
    train_dataset = HierarchicalPokerDataset(train_prompts, train_labels, tokenizer, max_length=max_length)
    val_dataset = HierarchicalPokerDataset(val_prompts, val_labels, tokenizer, max_length=max_length)
    test_dataset = HierarchicalPokerDataset(test_prompts, test_labels, tokenizer, max_length=max_length)

    # Get number of classes and class weights
    num_high_level_classes = train_dataset.get_num_high_level_classes()
    num_bet_classes = train_dataset.get_num_bet_classes()
    high_level_weights = train_dataset.get_high_level_weights()
    bet_weights = train_dataset.get_bet_weights()

    print(f"Number of high-level classes: {num_high_level_classes}")
    print(f"Number of bet-specific classes: {num_bet_classes}")

    # Load base model with minimal config to save memory
    print("Loading base model with memory optimizations...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "musketshugging/SFT225",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=token
    )

    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Create hierarchical classifier
    print("Initializing hierarchical classifier...")
    classifier = HierarchicalPokerClassifier(
        base_model,
        num_high_level_classes,
        num_bet_classes,
        high_level_weights.to(device),
        bet_weights.to(device)
    ).to(device)

    # Model has been successfully initialized
    model_loaded = True

    # Use smaller batch sizes and more accumulation steps to save memory
    training_args = TrainingArguments(
        output_dir="./hierarchical_poker_classifier",
        num_train_epochs=1,  # Reduced epochs for faster training
        per_device_train_batch_size=1,  # Smaller batch size
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,  # Less frequent evaluation
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="combined_accuracy",
        report_to="none",
        gradient_accumulation_steps=16,  # More accumulation steps
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,  # Avoid extra memory usage from workers
        remove_unused_columns=False,
        optim="adamw_torch",  # Use standard optimizer
        max_grad_norm=1.0,  # Clip gradients to prevent instability
    )

    # Get is_bet_sample for validation dataset
    val_is_bet_sample = val_dataset.is_bet_sample_tensor

    # Create custom trainer
    def compute_metrics_wrapper(pred):
        return compute_hierarchical_metrics(pred, val_is_bet_sample)

    trainer = HierarchicalTrainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_wrapper,
        is_bet_sample=val_is_bet_sample
    )

    try:
        # Train model
        print("Starting training...")
        trainer.train()

        # Evaluate on test set
        print("Evaluating on test set...")
        test_is_bet_sample = test_dataset.is_bet_sample_tensor

        def compute_test_metrics(pred):
            return compute_hierarchical_metrics(pred, test_is_bet_sample)

        trainer.compute_metrics = compute_test_metrics
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")

        # Save model
        print("Saving model...")
        trainer.save_model("./final_hierarchical_poker_classifier")

        # Save mappings
        high_level_mapping = train_dataset.get_high_level_mapping()
        bet_value_mapping = train_dataset.get_bet_value_mapping()

        mappings = {
            "high_level_mapping": high_level_mapping,
            "bet_value_mapping": bet_value_mapping
        }

        with open('./final_hierarchical_poker_classifier/label_mappings.json', 'w') as f:
            json.dump(mappings, f)

        print("Training complete!")
        return classifier, mappings

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("CUDA out of memory error. Try running with CPU or reducing dataset size.")
        else:
            print(f"Runtime error during training: {e}")

        # Clean up to free memory
        if 'classifier' in locals():
            del classifier
        if 'base_model' in locals():
            del base_model
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache()
        return None, None


if __name__ == "__main__":
    train_hierarchical_poker_model('pros_data1.json')