# study2

Overcooked 사용자 실험, HSP 기반 게임 실행, 사람형 에이전트/경로 예측 학습, Final Model 학습·평가 코드입니다.

## 설치

Python 3.9 환경에서 저장소와 HSP 정책 풀을 함께 받은 뒤 의존성을 설치합니다. 데이터와 모델은 Git LFS로 관리되므로 `git lfs pull`이 필요합니다.

```bash
git lfs install
git clone --recurse-submodules <repository-url>
cd study2
git lfs pull
python -m pip install -r requirements.txt
```

이미 clone한 저장소라면 정책 풀만 다음과 같이 받을 수 있습니다.

```bash
git submodule update --init --recursive
```

## 사용자 실험

베타 튜토리얼과 9-point calibration을 포함한 참가자 세션:

```bash
python main.py --tutorial --pid P01
```

중단된 본 실험 재개:

```bash
python main.py --pid P01 --trial completed
```

게임을 띄우지 않고 참가자별 27-trial 순서만 검증:

```bash
python main.py --pid P01 --dry-run --print-schedule
```

게임용 파트너는 `HSP`와 튜토리얼 연습용 `GREEDY`만 지원합니다. 실험 레이아웃은 `counter_circuit`, `unident_s_hard_onion`, `double_rings`입니다. 튜토리얼은 `cramped_room`, `bothway_coordination`, `coordination_ring`을 사용합니다.

## HSP 경로 예측 모델

정책 풀의 HSP·MEP rollouts로 데이터셋을 만들고 사람형 경로 예측 모델을 학습합니다.

```bash
python -m predict_path.hsp_clone.train_hsp_policy --help
python -m predict_path.hsp_clone.export_hsp_dataset --help
python -m predict_path.hsp_clone.rollout_hsp_clone --help
```

학습 결과는 `predict_path/output/`, 학습 데이터는
`data/model_training/predict_path/`에 둡니다.

## Final Model

Full/Partial 모델 및 EMMA 기반 gaze 설정을 포함한 학습 진입점:

```bash
python -m final_model.train --help
```

최종 선택 모델은 `final_model/checkpoint/final/`에 있습니다. 평가 코드는 `final_model/eval/`, 서버 병렬 학습 코드는 `final_model/server_parallel/`, Partial·EMMA 구성은 `final_model/partial_ver/`에 있습니다.

기본 테스트는 공개 최종 체크포인트만으로 실행됩니다.

```bash
python -m pytest -q
```

`final_model/server_parallel/tests`의 일부 통합 테스트는 공개 최종 세트에 포함하지 않은 과거 action source checkpoint와 CUDA 환경을 요구하므로 별도로 실행합니다.

## 데이터 및 공개 범위

본 저장소는 논문 출판 전 공개 버전으로, 연구 코드와 학습된 모델 체크포인트를 제공합니다.

참가자 P01–P36의 게임 로그, 원본 및 가공 시선 추적 데이터,
9-point 시선 보정 자료, 설문 응답, 참가자 단위 분석 데이터는
현재 공개 저장소에 포함하지 않습니다.

참가자 데이터는 승인된 연구 프로토콜에 따라 논문 출판 후
재식별 위험 검토와 비식별 처리를 거쳐 공개할 예정입니다.
여기에는 내부 참가자 식별자 교체와 자유서술 응답 검토가 포함됩니다.

따라서 참가자 단위 기록이 필요한 분석은 현재 공개 저장소만으로
완전히 재현할 수 없습니다.

데이터 공개 범위와 계획은
[data/DATASET_CARD.md](data/DATASET_CARD.md)를 참고하세요.

## 저장소 구성

- `main.py`, `tutorial.py`: 베타 튜토리얼, 9-point calibration, 본 실험 실행
- `predict_path/`: HSP rollout 데이터 생성과 사람형 경로 예측 모델 학습
- `final_model/`: Full/Partial/EMMA 학습, 평가 및 최종 체크포인트
- `user_study_analysis/`: 공개 데이터 분석 파이프라인
- `policy_pool/`: HSP 정책 풀 서브모듈
- `lib/overcooked_ai/`: 실험에서 사용하는 Overcooked 런타임

## Pre-publication Release and Data Availability

This repository provides the study's research code
and trained models.

Participant-level gameplay logs, gaze-tracking data,
questionnaire responses, and derived participant-level
datasets are not included before article publication.

The final IRB research protocol provides for the release
of anonymized participant data after article publication,
subject to the required privacy review.

Analyses requiring participant-level datasets cannot yet
be fully reproduced from this repository alone.

For details, see:

- [Data Availability](data/DATASET_CARD.md)
- [Research Code License](LICENSE)
- [Data and Documentation License](LICENSE-DATA.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)
