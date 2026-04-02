# Soft RL Baselines 수식 정리

## 문서 목적

이 문서는 `sql_base`, `sac_base`, `sql_s3`, `sac_s3`를 만들 때 사용한 이론적 정리를 남기기 위한 문서다.

핵심 질문은 세 가지다.

1. 우리가 맞추려는 posterior target은 정확히 무엇인가
2. 그 target을 `SQL`과 `SAC`로 어떻게 쓸 수 있는가
3. 이 관점이 `RTB`와 어떻게 연결되는가

아래 정리는 논문들의 결과를 현재 코드가 다루는

- autoregressive SMILES sequence
- terminal reward
- frozen prior LM

세팅에 맞게 풀어쓴 것이다.


## 공통 목표 분포

우리가 맞추려는 최종 target은

```text
pi*(x) ∝ pi0(x) * exp(beta * score(x))
```

이다.

여기서

- `x`: 최종 SMILES sequence
- `pi0(x)`: frozen pretrained prior LM이 주는 sequence 확률
- `score(x)`: terminal oracle score
- `beta`: score에 곱해지는 inverse temperature

중요한 점은 현재 코드에서는 `beta`를 reward에 미리 곱해 넣는 convention을 쓴다는 것이다.

즉 구현에서는

```text
r_terminal(x) = beta * score(x)
```

로 보고, soft backup 식 안에 temperature를 한 번 더 넣지 않는다.


## KL-regularized RL로 보는 관점

위 posterior target은 아래 최적화 문제와 같다.

```text
maximize_pi  E_{x ~ pi}[ beta * score(x) ] - KL(pi(x) || pi0(x))
```

sequence를 prefix 단위로 풀면

```text
maximize_pi  E_pi[
  beta * score(x) - sum_t log( pi(a_t | s_t) / pi0(a_t | s_t) )
]
```

즉 이 문제는

- `prior pi0`를 기준 policy로 둔
- trajectory-level KL regularized RL

로 볼 수 있다.


## SQL 전개

## local policy 형태

prior-regularized SQL에서는 state-value와 policy가 다음 형태를 가진다.

```text
V(s) = log sum_a pi0(a | s) exp(Q(s, a))
```

```text
pi(a | s) = pi0(a | s) exp(Q(s, a) - V(s))
```

이 식은 "prior로 shift된 softmax policy"라고 보면 된다.

현재 구현에서

- `prior_log_probs`가 `log pi0(a|s)`
- `correction_logits`가 `Q(s,a)` 역할

을 한다.


## terminal reward만 있는 경우

중간 reward가 없고 마지막 EOS transition에서만 reward가 들어가므로 Bellman target은

```text
y_t = V(s_{t+1})
```

```text
y_T = beta * score(x)
```

이다.

따라서 SQL critic regression은

```text
L_SQL = sum_t ( Q(s_t, a_t) - y_t )^2
```

가 된다.


## 코드 대응

현재 `sql_base` 구현의 핵심 부분은 다음과 같다.

파일:
- [run_sql_base.py](./main/smiles_gfn/run_sql_base.py)

```python
prior_log_probs = _masked_prior_log_probs(prior_logits, bos_token_id, pad_token_id)
combined_logits = prior_log_probs + correction_logits

next_soft_values = torch.logsumexp(combined_logits, dim=-1)
td_target[:, :-1] = next_soft_values[:, 1:]
td_target[batch_indices, terminal_indices] = beta * terminal_scores

td_error = (chosen_correction - td_target) ** 2
loss = (td_error * action_mask).sum() / action_mask.sum().clamp_min(1.0)
```

이 코드에서

- `chosen_correction` = sampled action의 `Q(s_t, a_t)`
- `logsumexp(prior + correction)` = `V(s_{t+1})`
- terminal target = `beta * score(x)`

로 대응된다.


## SQL과 posterior target의 관계

위 local equations를 trajectory로 telescope 하면

```text
log pi*(x) = log pi0(x) + beta * score(x) - const
```

가 되고, 결국

```text
pi*(x) ∝ pi0(x) * exp(beta * score(x))
```

를 얻는다.

즉 `SQL`은 local Bellman equation으로 우리가 원하는 posterior를 구현하는 방식이다.


## SAC 전개

`SAC`도 같은 posterior target을 쓸 수 있다.  
차이는 `policy`와 `Q`를 하나로 두지 않고 actor/critic을 분리한다는 점이다.


## critic target

prior-regularized discrete SAC에서는

```text
V(s) = sum_a pi(a | s) [ Q(s, a) - log( pi(a | s) / pi0(a | s) ) ]
```

를 쓴다.

terminal reward만 있을 때 critic target은

```text
y_t = V(s_{t+1})
```

```text
y_T = beta * score(x)
```

이다.

critic loss는

```text
L_critic = sum_t ( Q(s_t, a_t) - y_t )^2
```

이다.


## actor objective

actor는 각 state에서

```text
pi(. | s) ≈ argmin KL( pi(. | s) || const * pi0(. | s) * exp(Q(s, .)) )
```

를 하도록 업데이트된다.

이를 expectation 형태로 쓰면

```text
L_actor = sum_t E_{a ~ pi(. | s_t)}[
  log( pi(a | s_t) / pi0(a | s_t) ) - Q(s_t, a)
]
```

가 된다.


## 코드 대응

현재 `sac_base` 구현의 핵심 부분은 다음과 같다.

파일:
- [run_sac_base.py](./main/smiles_gfn/run_sac_base.py)

```python
combined_logits = prior_log_probs + policy_correction_logits
policy_log_probs = F.log_softmax(combined_logits, dim=-1)
policy_probs = policy_log_probs.exp()

next_state_values = (
    policy_probs.detach() * (target_q_values - detached_log_ratio)
).sum(dim=-1)

td_target[:, :-1] = next_state_values[:, 1:]
td_target[batch_indices, terminal_indices] = beta * terminal_scores

critic_loss = ((chosen_q_values - td_target) ** 2 * action_mask).sum() / action_mask.sum().clamp_min(1.0)

actor_state_loss = (
    policy_probs * (safe_log_ratio - detached_q_values)
).sum(dim=-1)
actor_loss = (actor_state_loss * action_mask).sum() / action_mask.sum().clamp_min(1.0)
```

여기서

- `safe_log_ratio` = `log pi - log pi0`
- `next_state_values` = `V(s_{t+1})`
- `chosen_q_values` = sampled action의 critic 값

이다.


## 왜 base에서는 SQL이 더 유리할 수 있는가

현재 코드 기준으로 `sql_base`와 `sac_base`의 가장 큰 차이는 다음이다.

- `SQL`: policy correction과 Q가 사실상 같은 객체
- `SAC`: actor와 critic을 분리해서 둘 다 근사해야 함

on-policy + sparse terminal reward setting에서는 후자가 더 어렵다.

즉

- `SQL`은 같은 posterior target을 더 직접적으로 맞추고
- `SAC`는 더 일반적이지만 critic approximation burden이 추가된다

고 볼 수 있다.


## RTB와의 관계

## RTB가 보는 문제

`RTB`는 generative model prior와 terminal reward가 있을 때 posterior inference 문제

```text
p_post(x) ∝ p0(x) * exp(beta * score(x))
```

를 직접 다루는 관점이다.

즉 target posterior 자체는 위에서 정의한 soft RL target과 동일한 family다.


## TB / RTB vs SQL / SAC

관계를 짧게 정리하면 이렇다.

- `TB`, `RTB`: trajectory 전체에 대한 global consistency
- `SQL`, `SAC`, `PCL`, `Trust-PCL`: state/action 수준의 local consistency

즉 차이는 "무슨 posterior를 목표로 하느냐"보다
"그 posterior를 어떤 consistency equation으로 학습시키느냐"에 가깝다.


## 현재 세팅에서의 해석

우리 세팅은

- autoregressive sequence
- terminal reward
- prior LM이 주어진 fine-tuning

이다.

이 경우 아래 해석이 자연스럽다.

1. `RTB`는 posterior `pi0(x) * exp(beta * score(x))`를 trajectory-level equation으로 맞춘다.
2. `SQL` / `SAC`는 같은 posterior를 local Bellman / local KL projection으로 맞춘다.
3. 따라서 sequence fine-tuning 문제에서는 둘이 매우 가까운 target을 공유한다.

정확히 말하면,

- `RTB`는 `full trajectory constraint`
- `SQL/SAC`는 `local constraint`

를 쓴다.


## 왜 `sql_s3`와 `sac_s3`가 `run.py` 구조를 그대로 따라가도 되는가

이론적으로 보면 `S3`의 replay/negative replay/auxiliary loss는
"posterior target"을 바꾸는 것이 아니라
"어떤 데이터로, 얼마나 off-policy하게 학습하느냐"를 바꾸는 것이다.

그래서 구현에서는

- `sql_s3`: 기존 `run.py` 구조 유지 + SQL objective
- `sac_s3`: 기존 `run.py` 구조 유지 + SAC objective

처럼 분리해도 자연스럽다.


## 구현 관점 요약

### `sql_base`

- on-policy
- prior-shifted local soft Bellman
- 가장 직접적인 posterior fitting

### `sac_base`

- on-policy
- actor/critic 분리
- 같은 posterior target이지만 더 어려운 optimization

### `sql_s3`

- `run.py`의 S3 replay machinery 사용
- objective만 SQL

### `sac_s3`

- `run.py`의 S3 replay machinery 사용
- objective만 SAC


## 참고한 1차 자료

아래 자료를 참고했다.

- Trajectory Balance: Improved Credit Assignment in GFlowNets
  https://arxiv.org/abs/2201.13259
- Soft Q-Learning
  https://arxiv.org/abs/1702.08165
- Path Consistency Learning
  https://arxiv.org/abs/1702.08892
- Trust-PCL
  https://arxiv.org/abs/1707.01891
- Soft Actor-Critic
  https://arxiv.org/abs/1801.01290
- Amortizing intractable inference in diffusion models for vision, language, and control
  https://openreview.net/forum?id=gVTkMsaaGI
- Relative Trajectory Balance is equivalent to Trust-PCL
  https://openreview.net/forum?id=Ykat0gfqfM


## 해석 범위 메모

마지막으로, 아래 문장은 논문 문장을 그대로 옮긴 것이 아니라
위 자료들을 현재 SMILES fine-tuning setting에 맞게 정리한 해석이다.

```text
autoregressive terminal-reward setting에서는
SQL / SAC / PCL / RTB가 같은 posterior family를 공유하고,
차이는 local constraint냐 trajectory constraint냐에 있다.
```

이 해석은 현재 구현 방향을 설명하기 위한 것이고,
strict한 정리는 각 논문의 정확한 가정 위에서 읽는 것이 맞다.
