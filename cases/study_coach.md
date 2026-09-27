# Agent Study Coach — 구현 성공 뒤에서 발견한 평가 오염

## 구현 사실

- Day 1 PDF 6쪽 + Day 2 PDF 6쪽 + Day 3 Markdown 1개 = 원본 단위 13개
- chunk size 800, overlap 120 → 60 chunks
- OpenAI `text-embedding-3-small`, Chroma, top-k 4
- 도구 2개: 노트 검색, 복습 퀴즈
- LangGraph 흐름: agent → tools → agent
- `thread_id`별 InMemorySaver 세션 기억

## 발견한 실패

시스템은 노트에 없는 “양자역학 슈뢰딩거 방정식” 질문에 `[근거 없음]`을 반환하도록 설계됐습니다. 그러나 실제 실행에서는 검색 결과가 나왔습니다. Day 3 보고서 자체가 그 평가 질문과 성공 설명을 포함한 채 지식 corpus에 색인됐기 때문입니다. 시스템은 원래 학습노트가 아니라 자기 평가 보고서를 근거로 인식했습니다.

또한 “Day 2 Chain vs Agent” 검색도 원본 Day 2 노트보다 Day 3 보고서의 요약·캡처를 먼저 반환했습니다. 정확한 단어가 하나라도 겹치면 통과시키는 기존 no-evidence 규칙은 의미적 근거 부족을 잡지 못합니다.

## 수정 설계

1. `knowledge/`와 `evaluation/`, `reports/` 디렉터리를 물리적으로 분리합니다.
2. 검색기는 `knowledge` collection만 조회하게 합니다.
3. 출처 문서 유형을 metadata로 강제하고 보고서 유형은 차단합니다.
4. 답변 근거를 인용 span과 함께 검증하는 별도 evaluator를 둡니다.
5. paraphrase·외부 질문·충돌 문서가 포함된 독립 평가셋을 만듭니다.

## 주장 범위

구현과 실행은 확인했지만 실제 사용자 활용은 없고 학습 향상도 측정하지 않았습니다. LangSmith trace의 실행 성공은 답변 품질이나 사용자 효과와 같지 않습니다.
