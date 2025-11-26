
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


def option_maker(row):
    A = row['A']
    B = row['B']
    C = row['C']
    D = row['D']
    E = row['E']
    return f"1) {A}\n2) {B}\n3) {C}\n4) {D}\n5) {E}"

def get_sean0042_KorMedMCQA_prompt(row_sample):
    text = (
        "당신은 의사입니다.\n"
        f"질문:\n{row_sample['question']}\n\n"
        f"보기:\n{option_maker(row_sample)}\n\n"
        "질문을 분석하고, 제시된 보기 중에서 한개의 보기를 선택하고, 근거를 설명해.\n"
        "- JSON 외의 어떤 추가 텍스트도 출력하지 마세요."
        "- 반드시 다음 JSON 형식으로만 답하세요:\n"
        '{"answer":"보기","explanation":"한국어 근거"}\n'
    )
    return text