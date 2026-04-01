# 브랜치 전략

## 브랜치 구조
- main : 최종 완성본만, 직접 커밋 절대 금지
- dev  : 팀원 작업 통합 브랜치
- 작업 흐름: 개발 브랜치 → dev PR → main PR

## 브랜치 네이밍
형식: prefix/기능명

prefix 종류:
- feature : 새로운 기능 개발
- fix     : 버그 수정
- hotfix  : 긴급 버그 수정

예시:
- feature/adaptive-loss
- feature/data-pipeline
- feature/kupiec-test
- fix/vix-date-alignment

---

# 커밋 형식
```
타입: 작업내용
```

## 타입 종류
- feat: 새로운 기능
- fix: 버그 수정
- data: 데이터 수집·전처리
- model: 모델 코드
- val: 검증·테스트
- docs: 문서
- refactor: 리팩토링
- chore: 설정·패키지

## 커밋 예시
```
model: M3 Adaptive Pinball Loss VIX·σ 연동 구현
model: M3 Multi-Quantile Head Q0.1·Q0.5·Q0.9 구성
model: M3 Optuna 하이퍼파라미터 탐색 추가
val: Kupiec POF Test 코드 구현
fix: VIX 데이터 날짜 정렬 오류 수정
```

---

# 커밋 규칙
- 한 커밋에 하나의 작업만
- main 직접 커밋 절대 금지
- PR 없이 dev merge 금지
