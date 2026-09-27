# sLLM vs RAG — 무엇을 증명했고 무엇을 증명하지 못했는가

## 구현 사실

- Qwen2.5-0.5B-Instruct: 총 495,114,112 parameters
- LoRA: q/k/v/o projection, rank 8, alpha 16, dropout 0.05
- 학습 가능 parameter: 1,081,344개(0.2184%)
- 합성 Q&A 8개, 5 epochs, optimizer 10 steps
- 평가 질문 5개(A 2, B 2, C 1)
- RAG: Sentence-Transformers + FAISS

## 관찰

RAG는 외부 문서의 CUDA 12.4를 회수했고, 문서 값을 13.0으로 바꾼 실험에서도 새 값을 답했습니다. LoRA는 말투·형식의 일부 변화는 보였지만 CUDA 값을 10.2로 답했습니다. 고정 top-k=2는 유사도 0.437의 무관 문서도 가져와 답변을 오염시킬 수 있었습니다.

## 결정적 한계

두 B 질문은 학습 Q&A 및 지식 문서의 질문과 동일했습니다. 높은 유사도와 정답은 독립적인 일반화 성능이 아니라 exact-match 누수의 영향을 받습니다. 검증 분할, 반복 실행, 정량적 생성 평가도 없습니다.

## 결론

이 결과로 “RAG가 항상 우월하다”고 말할 수 없습니다. 제한된 교육용 조건에서 외부 사실을 검색·교체할 수 있음을 확인했으며, 다음 실험에는 paraphrase/반례/미답 문항으로 구성된 독립 평가셋과 retrieval Recall@k, answer faithfulness가 필요합니다.
