def calculator(expression: str) -> str:
    try:
        # 위험한 eval 대신 안전한 연산만 허용
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"계산 오류: {str(e)}"