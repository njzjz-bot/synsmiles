# Soft RL Baselines 개발 설명 및 사용 방법

## 개요

이 브랜치에서는 `smiles_gfn` 실험 코드 위에 4개의 soft RL baseline을 추가했다.

- `sql_base`
- `sql_s3`
- `sac_base`
- `sac_s3`

핵심 목표는 다음 두 축을 비교할 수 있게 만드는 것이다.

- objective 축:
  `SQL` vs `SAC`
- data regime 축:
  `base` vs `S3`

여기서 posterior target은 공통으로

```text
pi*(x) ∝ prior(x) * exp(beta * score(x))
```

를 따른다.  
`prior`는 frozen pretrained SMILES LM이고, `score(x)`는 PMO oracle 점수다.


## 각 method의 의미

### 1. `sql_base`

파일:
- [run_sql_base.py](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/run_sql_base.py)
- [hparams_sql_base.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sql_base.yaml)

의도:
- `reinvent_rs` 스타일의 기본 실험 골격을 유지
- objective만 prior-regularized SQL로 교체

특징:
- on-policy
- `reshape_reward: True`가 기본
- 합성 가능 샘플만 oracle 점수를 받고, 나머지는 0 reward
- replay 없음


### 2. `sac_base`

파일:
- [run_sac_base.py](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/run_sac_base.py)
- [hparams_sac_base.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sac_base.yaml)

의도:
- `sql_base`와 같은 base setting에서 SAC objective 비교

특징:
- on-policy
- `Shared pretrained backbone + separate actor/critic heads`
- critic target network 사용
- `reshape_reward: True`가 기본

구현 메모:
- actor는 `prior + correction logits` 형태로 policy를 만든다.
- critic은 vocab-sized `Q(s,a)`를 예측한다.
- `loss = actor_loss + critic_loss`


### 3. `sql_s3`

파일:
- [run_sql_s3.py](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/run_sql_s3.py)
- [hparams_sql_s3.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sql_s3.yaml)

의도:
- 기존 `run.py`의 S3 machinery를 최대한 유지
- sampler와 main objective만 SQL로 교체

특징:
- off-policy replay 사용
- positive replay / negative replay 모두 사용
- `filter_unsynthesizable`, `aux_loss`, negative sampling, GA path를 기존 구조에 맞게 유지


### 4. `sac_s3`

파일:
- [run_sac_s3.py](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/run_sac_s3.py)
- [hparams_sac_s3.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sac_s3.yaml)

의도:
- `sql_s3`의 off-policy machinery를 그대로 재사용
- objective만 SAC로 교체

특징:
- on-policy / replay 모드 전환 구조는 `sql_s3`와 동일
- negative replay, auxiliary loss, replay sampling 정책도 동일
- target critic soft update 사용


## 구현 구조

### 엔트리 포인트

런처는 [run.py](/home/mila/m/minsu.kim/synsmiles/src/pmo/run.py) 에 연결되어 있다.

추가된 method:

- `sql_base`
- `sac_base`
- `sql_s3`
- `sac_s3`


### 보조 수정

- [genetic_operator/mutate.py](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/genetic_operator/mutate.py)
  오래된 import 경로를 현재 트리에 맞게 수정했다.


## 실행 방법

### 기본 위치

```bash
cd /home/mila/m/minsu.kim/synsmiles/src/pmo
```

Python 환경은 기존 실험과 동일하게 아래 환경을 사용했다.

```bash
/home/mila/m/minsu.kim/envs/rxnflow/bin/python
```


### 1k helper script

다음 스크립트를 바로 사용할 수 있다.

- [run_sql_base_1k.sh](/home/mila/m/minsu.kim/synsmiles/src/pmo/run_sql_base_1k.sh)
- [run_sac_base_1k.sh](/home/mila/m/minsu.kim/synsmiles/src/pmo/run_sac_base_1k.sh)
- [run_sql_s3_1k.sh](/home/mila/m/minsu.kim/synsmiles/src/pmo/run_sql_s3_1k.sh)
- [run_sac_s3_1k.sh](/home/mila/m/minsu.kim/synsmiles/src/pmo/run_sac_s3_1k.sh)

예:

```bash
bash /home/mila/m/minsu.kim/synsmiles/src/pmo/run_sql_base_1k.sh
```


### 직접 실행

#### SQL base

```bash
/home/mila/m/minsu.kim/envs/rxnflow/bin/python run.py sql_base \
  --oracles drd2 \
  --wandb disabled \
  --run_name sql_base_drd2 \
  --config_default hparams_sql_base.yaml \
  --seed 0 \
  --max_oracle_calls 1000 \
  --freq_log 100
```

#### SAC base

```bash
/home/mila/m/minsu.kim/envs/rxnflow/bin/python run.py sac_base \
  --oracles drd2 \
  --wandb disabled \
  --run_name sac_base_drd2 \
  --config_default hparams_sac_base.yaml \
  --seed 0 \
  --max_oracle_calls 1000 \
  --freq_log 100
```

#### SQL S3

```bash
/home/mila/m/minsu.kim/envs/rxnflow/bin/python run.py sql_s3 \
  --oracles drd2 \
  --wandb disabled \
  --run_name sql_s3_drd2 \
  --config_default hparams_sql_s3.yaml \
  --seed 0 \
  --max_oracle_calls 1000 \
  --freq_log 100
```

#### SAC S3

```bash
/home/mila/m/minsu.kim/envs/rxnflow/bin/python run.py sac_s3 \
  --oracles drd2 \
  --wandb disabled \
  --run_name sac_s3_drd2 \
  --config_default hparams_sac_s3.yaml \
  --seed 0 \
  --max_oracle_calls 1000 \
  --freq_log 100
```


## 설정 파일 설명

### Base 설정

- [hparams_sql_base.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sql_base.yaml)
- [hparams_sac_base.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sac_base.yaml)

현재 기본값:

- `reshape_reward: True`
- `use_retrosynthesis: True`
- `retro_env: stock_hb`
- `beta: 50`

즉 base는 "합성 가능성 기반 reward shaping이 켜진 on-policy baseline"이다.


### S3 설정

- [hparams_sql_s3.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sql_s3.yaml)
- [hparams_sac_s3.yaml](/home/mila/m/minsu.kim/synsmiles/src/pmo/main/smiles_gfn/hparams_sac_s3.yaml)

주요 파라미터:

- `num_keep`: replay buffer 크기
- `experience_loop`: on-policy / replay 전환 주기
- `experience_replay`: replay에서 샘플할 trajectory 수
- `filter_unsynthesizable`: 합성 불가능 샘플을 학습 대상에서 제외할지 여부
- `aux_loss`: negative replay 기반 보조 loss 종류
- `target_tau`: SAC target critic soft update 계수


## 메모리 주의사항

현재 기본 S3 설정은 메모리 사용량이 큰 편이다.

- `batch_size: 64`
- `experience_replay: 64`

실제 실행 중 일부 GPU 노드에서는 `sql_s3` / `sac_s3` 400-call 실험 도중 CUDA OOM이 났다.  
이 경우 가장 먼저 줄일 값은 아래 두 개다.

- `batch_size`
- `experience_replay`

필요하면 `max_length`도 함께 줄이는 편이 안전하다.


## 로그 해석

### SQL 계열

주요 로그:

- `kl`
- `log_ratio`
- `prior_lp`

의미:

- `kl`: 현재 policy가 prior에서 얼마나 벗어났는지
- `log_ratio`: 샘플된 action 기준의 `log pi - log prior`
- `prior_lp`: prior가 선택한 action에 준 평균 log-prob


### SAC 계열

주요 로그:

- `actor`
- `critic`
- `kl`
- `log_ratio`
- `prior_lp`

의미:

- `actor`: policy update loss
- `critic`: Bellman target에 대한 critic regression loss
- 나머지는 SQL과 동일한 prior-regularization 해석


## 현재 관찰 메모

- `sql_base`는 on-policy base에서 안정적으로 동작했다.
- `sac_base`는 동작은 하지만 현재 base setting에서는 `sql_base`보다 일관되게 강하지는 않았다.
- `sac_s3`는 `sql_s3`와 거의 같은 구조에서 돌아가며, 작은 S3 설정 기준으로는 `sql_s3`와 비슷하거나 약간 나은 결과가 나왔다.
- 다만 `sac_s3`는 critic loss가 더 noisy하므로, 큰 설정에서는 메모리와 안정성을 같이 확인하는 것이 좋다.


## 권장 사용 순서

처음 실험할 때는 아래 순서를 권장한다.

1. `sql_base`
2. `sac_base`
3. `sql_s3`
4. `sac_s3`

이 순서가 가장 디버깅이 쉽고, 문제 발생 시 원인을 분리하기 좋다.
