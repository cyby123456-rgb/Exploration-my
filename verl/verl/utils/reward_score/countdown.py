import re
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Prefer explicit <answer> tags when present.
    answer = _extract_answer_from_text(solution_str)
    if answer is not None:
        return answer

    # Fallback to parsing content after Assistant markers, or the full string if absent.
    if "Assistant:" in solution_str:
        tail = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        tail = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        tail = solution_str

    return _extract_equation_fallback(tail)


def _extract_answer_from_text(text):
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _extract_equation_fallback(text):
    candidates = list(re.finditer(r"[\d+\-*/().=\s]+", text))
    for match in reversed(candidates):
        candidate = match.group(0).strip()
        if not candidate:
            continue
        if not re.search(r"\d", candidate):
            continue
        if not re.search(r"[+\-*/]", candidate):
            continue
        if "=" in candidate:
            candidate = candidate.split("=", 1)[0].strip()
        if candidate:
            return candidate
    return None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        if re.search(r"\d+\.\d+", equation_str):
            return False
        if re.search(r"[eE]", equation_str):
            return False
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using a restricted AST."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        tree = ast.parse(equation_str, mode="eval")
        return _safe_eval(tree)
    except Exception:
        return None


def _safe_eval(node):
    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
        return unary_ops[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
        return bin_ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError("Disallowed expression")


def compute_score(solution_str, ground_truth, method='strict', format_score=0.0, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Target: {target} | Numbers: {numbers}")
    #     print(f"Extracted equation: {equation}")
    #     print(f"Solution string: {solution_str}")

    if equation is None:
        # if do_print:
        #     print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        # if do_print:
        #     print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            # if do_print:
            #     print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            # if do_print:
            #     print(f"Correct equation: {equation} = {result}")
            return score
        else:
            # if do_print:
            #     print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        # if do_print:
        #     print(f"Error evaluating equation")
        return format_score 
    
def compute_score_for_eval(solution_str, ground_truth, method='strict'):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer

    Return:
        is_correct_strict:
        reward_score:
        is_correct_format:
        is_correct_finalanswer:
    """
    format_score=0.0
    score=1.
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Target: {target} | Numbers: {numbers}")
    #     print(f"Extracted equation: {equation}")
    #     print(f"Solution string: {solution_str}")

    if equation is None:
        # if do_print:
        #     print(f"⭐️No equation found")
        return 0, 0, 0, 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        # if do_print:
        #     print(f"⭐️Invalid equation")
        return 0, format_score, 0, 0
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            # if do_print:
            #     print(f"⭐️Could not evaluate equation")
            return 0, format_score, 0, 0
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            # if do_print:
            #     print(f"⭐️⭐️⭐️Correct equation: {equation} = {result}")
            return 1, 1, 1, 1
        else:
            # if do_print:
            #     print(f"⭐️Wrong result: equation = {result}, target = {target}")
            return 0, format_score, 1, 0
    except:
        # if do_print:
        #     print(f"⭐️Error evaluating equation")
        return 0, format_score, 0, 0
