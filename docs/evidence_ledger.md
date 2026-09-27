# 공개 주장 원장

| 주장 | 상태 | 공개 근거 |
|---|---|---|
| SECOM 데이터·분할·holdout 지표 | 검증됨 | `results/final_seed42_validation.json` |
| Recall bootstrap 95% CI 76.2–100.0% | 검증됨, 표본 작음 | 동일 JSON |
| 상위 80개 센서 우월 | 반증되어 철회 | `results/repeated_feature_count_summary.json` |
| RAG가 외부 값을 회수·교체 | 제한된 관찰 | `cases/rag_sllm.md` |
| RAG 일반 우월성 | 증명되지 않음 | 독립 평가셋 부재 |
| Study Coach 구현 | 검증됨 | `cases/study_coach.md` |
| Study Coach 실제 사용자 효과 | 주장하지 않음 | 사용 이력·전후 평가 없음 |

원본 notebook·보고서에는 개인 경로와 API 사용 흔적이 포함될 수 있어 공개 저장소에 올리지 않았습니다. 대신 공개 가능한 재현 코드와 수치 결과, 제한 사항을 제공합니다.
