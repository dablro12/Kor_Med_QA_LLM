
import ast
def get_snuh_ClinicalQA_prompt(row_sample):
    # Parse options safely (assuming it's a string representation of a dict)

    # options 파싱 및 포맷팅
    options_dict = ast.literal_eval(row_sample['options'])
    formatted_options = "\n".join(
        f"{key.replace('option_', '').upper()}) {value}"
        for key, value in sorted(options_dict.items())
    )

    text = (
        "당신은 의사입니다.\n"
        f"질문:\n{row_sample['question']}\n\n"
        f"보기:\n{formatted_options}\n\n"
        "질문을 분석하고, 제시된 보기 중에서 정답을 선택해.\n"
        "- JSON 외의 어떤 추가 텍스트도 출력하지 마세요."
        "- 정답을 다음 JSON 형식으로만 답하세요:\n"
        '{"answer":"보기","explanation":"한국어 근거"}'
    )


    return text