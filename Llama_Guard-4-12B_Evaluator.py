import json
import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from typing import List, Dict
import argparse
from tqdm import tqdm

class LlamaGuardChecker:
    def __init__(self, model_id: str = "meta-llama/Llama-Guard-4-12B"):
        """Initialize Llama Guard 4 model and processor."""
        print(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        print("Model loaded successfully!")
    
    def check_safety(self, text: str, role: str = "assistant", user_prompt: str = None) -> Dict[str, str]:
        """
        Check safety of a single text.
        
        Args:
            text: The text to check
            role: Either 'user' (for input/prompt) or 'assistant' (for response/output)
            user_prompt: Required when role='assistant', the user's question that prompted the response
        
        Returns:
            Dict with 'text', 'safety_label', and 'categories' (if unsafe)
        """
        if role == "assistant":
            # For assistant responses, we need the full conversation
            if user_prompt is None:
                raise ValueError("user_prompt is required when checking assistant responses")
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}]
                }
            ]
        else:
            # For user prompts, just check the prompt itself
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}]
                }
            ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
        
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )[0]
        
        # Parse response
        lines = response.strip().split('\n')
        safety_label = lines[0].strip()
        categories = lines[1:] if len(lines) > 1 else []
        
        return {
            "text": text,
            "safety_label": safety_label,
            "categories": categories if categories else []
        }
    
    def check_conversation(self, question: str, answer: str) -> Dict:
        """
        Check safety of both question (input) and answer (output).
        
        Args:
            question: The user question/prompt
            answer: The model response/answer
        
        Returns:
            Dict with safety results for both question and answer
        """
        question_result = self.check_safety(question, role="user")
        answer_result = self.check_safety(answer, role="assistant", user_prompt=question)
        
        return {
            "question": question,
            "answer": answer,
            "question_safety": question_result,
            "answer_safety": answer_result,
            "overall_safe": (question_result["safety_label"] == "safe" and 
                           answer_result["safety_label"] == "safe")
        }

def process_json_file(input_file: str, output_file: str, 
                      question_key: str = "question", 
                      answer_key: str = "answer",
                      check_questions: bool = True,
                      check_answers: bool = True):
    """
    Process a JSON file containing questions and answers.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file with safety results
        question_key: Key name for questions in JSON (default: "question")
        answer_key: Key name for answers in JSON (default: "answer")
        check_questions: Whether to check question safety (default: True)
        check_answers: Whether to check answer safety (default: True)
    """
    # Load JSON data
    print(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list of dicts and single dict
    if isinstance(data, dict):
        data = [data]
    
    # Initialize checker
    checker = LlamaGuardChecker()
    
    # Process each item
    results = []
    print(f"Processing {len(data)} items...")
    
    for item in tqdm(data):
        question = item.get(question_key, "")
        answer = item.get(answer_key, "")
        
        result = {
            "original": item,
            "question": question,
            "answer": answer
        }
        
        if check_questions and question:
            result["question_safety"] = checker.check_safety(question, role="user")
        
        if check_answers and answer:
            result["answer_safety"] = checker.check_safety(answer, role="assistant", user_prompt=question)
        
        # Determine overall safety
        q_safe = not check_questions or result.get("question_safety", {}).get("safety_label") == "safe"
        a_safe = not check_answers or result.get("answer_safety", {}).get("safety_label") == "safe"
        result["overall_safe"] = q_safe and a_safe
        
        results.append(result)
    
    # Save results
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total = len(results)
    unsafe_count = sum(1 for r in results if not r["overall_safe"])
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total items processed: {total}")
    print(f"Safe items: {total - unsafe_count}")
    print(f"Unsafe items: {unsafe_count}")
    print(f"Safety rate: {(total - unsafe_count) / total * 100:.2f}%")
    
    if check_questions:
        q_unsafe = sum(1 for r in results 
                      if r.get("question_safety", {}).get("safety_label") != "safe")
        print(f"\nUnsafe questions: {q_unsafe}")
    
    if check_answers:
        a_unsafe = sum(1 for r in results 
                      if r.get("answer_safety", {}).get("safety_label") != "safe")
        print(f"Unsafe answers: {a_unsafe}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Llama Guard 4 safety checks on JSON question-answer data"
    )
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="safety_results.json",
        help="Path to output JSON file (default: safety_results.json)"
    )
    parser.add_argument(
        "--question_key", 
        type=str, 
        default="question",
        help="Key name for questions in JSON (default: question)"
    )
    parser.add_argument(
        "--answer_key", 
        type=str, 
        default="answer",
        help="Key name for answers in JSON (default: answer)"
    )
    parser.add_argument(
        "--skip_questions",
        action="store_true",
        help="Skip safety checking of questions (only check answers)"
    )
    parser.add_argument(
        "--skip_answers",
        action="store_true",
        help="Skip safety checking of answers (only check questions)"
    )
    
    args = parser.parse_args()
    
    process_json_file(
        input_file=args.input_file,
        output_file=args.output_file,
        question_key=args.question_key,
        answer_key=args.answer_key,
        check_questions=not args.skip_questions,
        check_answers=not args.skip_answers
    )