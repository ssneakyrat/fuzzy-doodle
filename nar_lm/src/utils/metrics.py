# src/utils/metrics.py
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import time

class LanguageModelMetrics:
    """Evaluation metrics for language models"""
    
    @staticmethod
    def perplexity(loss):
        """Calculate perplexity from loss"""
        return torch.exp(loss).item()
    
    @staticmethod
    def sequence_accuracy(predictions, targets, pad_token_id):
        """Calculate sequence-level accuracy"""
        # Create mask to ignore padding
        mask = (targets != pad_token_id)
        
        # Calculate accuracy only on non-padded tokens
        correct = ((predictions == targets) & mask).sum().item()
        total = mask.sum().item()
        
        return correct / total if total > 0 else 0
    
    @staticmethod
    def token_accuracy(predictions, targets, pad_token_id):
        """Calculate token-level accuracy (percentage of sentences with all tokens correct)"""
        # Mask for padding
        mask = (targets != pad_token_id)
        
        # Check if all non-padded tokens in each sequence match
        seq_correct = torch.all((predictions == targets) | ~mask, dim=1).sum().item()
        total_seqs = targets.size(0)
        
        return seq_correct / total_seqs
    
    @staticmethod
    def calculate_bleu(references, hypothesis, smoothing=True):
        """Calculate BLEU score for generated text"""
        if smoothing:
            smoothie = SmoothingFunction().method1
            return sentence_bleu([references], hypothesis, smoothing_function=smoothie)
        else:
            return sentence_bleu([references], hypothesis)
    
    @staticmethod
    def calculate_rouge(reference, hypothesis):
        """Calculate ROUGE scores for generated text"""
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]
    
    @staticmethod
    def latent_diversity(latent_vectors):
        """Measure diversity in latent space representations"""
        # Flatten latent vectors
        flattened = latent_vectors.reshape(latent_vectors.size(0), -1)
        
        # Calculate pairwise cosine similarity
        norm = torch.norm(flattened, dim=1, keepdim=True)
        normalized = flattened / norm
        similarity_matrix = torch.matmul(normalized, normalized.transpose(0, 1))
        
        # Calculate diversity score (1 - average similarity)
        diversity = 1.0 - (torch.sum(similarity_matrix) - torch.trace(similarity_matrix)) / (latent_vectors.size(0) * (latent_vectors.size(0) - 1))
        
        return diversity.item()
    

class SpeedMetrics:
    """Metrics for measuring generation speed"""
    
    @staticmethod
    def measure_generation_time(model, tokenizer, prompts, device, max_length=None):
        """Measure generation time for a list of prompts"""
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        total_time = 0
        total_tokens = 0
        results = []
        
        for prompt in prompts:
            # Tokenize input
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            input_length = input_ids.size(1)
            
            # Measure generation time
            start_time = time.time()
            
            with torch.no_grad():
                output_ids = model.generate(input_ids, max_length=max_length)
                
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Calculate token count and speed
            output_length = output_ids.size(1)
            generated_tokens = output_length - input_length
            total_tokens += generated_tokens
            total_time += generation_time
            
            # Decode output
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Store result
            results.append({
                "prompt": prompt,
                "output": output_text,
                "time": generation_time,
                "tokens": generated_tokens,
                "tokens_per_second": generated_tokens / generation_time if generation_time > 0 else 0
            })
        
        # Calculate average speed
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "results": results,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "avg_tokens_per_second": avg_tokens_per_second
        }
    
    @staticmethod
    def compare_generation_speed(nar_model, ar_model, tokenizer, prompts, device, max_length=None):
        """Compare generation speed between NAR and AR models"""
        nar_metrics = SpeedMetrics.measure_generation_time(nar_model, tokenizer, prompts, device, max_length)
        ar_metrics = SpeedMetrics.measure_generation_time(ar_model, tokenizer, prompts, device, max_length)
        
        # Calculate speedup
        speedup = ar_metrics["total_time"] / nar_metrics["total_time"] if nar_metrics["total_time"] > 0 else float('inf')
        
        return {
            "nar_metrics": nar_metrics,
            "ar_metrics": ar_metrics,
            "speedup": speedup
        }


class RefinementMetrics:
    """Metrics for evaluating the refinement process"""
    
    @staticmethod
    def track_refinement_progress(model, tokenizer, prompt, device, max_length=None):
        """Track refinement progress for a given prompt"""
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Track initial generation and refinements
        results = []
        
        # Initial encoding
        encoder_output = model.encoder(input_ids)
        latent_output, _ = model.latent_mapper(encoder_output)
        
        # Initial prediction
        logits = model.decoder(latent_output)
        preds = torch.argmax(logits, dim=-1)
        
        # Record initial prediction
        output_text = tokenizer.decode(preds[0], skip_special_tokens=True)
        results.append({
            "step": 0,
            "output": output_text,
            "confidence": torch.mean(torch.max(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[0]).item()
        })
        
        # Iterative refinement
        for step in range(model.config.num_refinement_steps):
            # Re-encode with the predicted tokens
            encoder_output = model.encoder(preds)
            latent_output, _ = model.latent_mapper(encoder_output)
            
            # Get new predictions
            logits = model.decoder(latent_output, encoder_output)
            new_preds = torch.argmax(logits, dim=-1)
            
            # Calculate confidence
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            avg_confidence = torch.mean(confidence).item()
            
            # Update high-confidence predictions
            confidence_mask = confidence > model.config.confidence_threshold
            preds = torch.where(confidence_mask, new_preds, preds)
            
            # Record current prediction
            output_text = tokenizer.decode(preds[0], skip_special_tokens=True)
            results.append({
                "step": step + 1,
                "output": output_text,
                "confidence": avg_confidence,
                "changed_tokens": torch.sum(~confidence_mask).item()
            })
        
        return results