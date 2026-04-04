# Математические заметки: Logit Jump-Diffusion ядро для Polymarket

## Концепт

В prediction markets до сих пор нет стандартного унифицированного pricing/risk-фреймворка уровня Black-Scholes для опционов.

Этот крейт реализует подход **Logit Jump-Diffusion**, описанный в:

- Shaw & Dalen (2025), *Toward Black-Scholes for Prediction Markets: A Unified Kernel and Market-Maker's Handbook*

Цель прикладная и HFT-ориентированная: стабильно и быстро строить inventory-aware bid/ask-котировки в реальном времени по большому числу рынков и давать низколатентную аналитику для принятия решений.

## Почему logit-пространство

На Polymarket цены выражены как вероятности:

$$
p \in (0,1)
$$

Считать диффузию напрямую в probability-space неудобно, потому что 0 и 1 являются жёсткими границами.

Мы переводим вероятность в log-odds (logit):

$$
x = \log\left(\frac{p}{1-p}\right), \quad p = S(x)=\frac{1}{1+e^{-x}}
$$

Теперь переменная состояния живёт на всей вещественной прямой:

$$
x \in (-\infty, +\infty)
$$

Это позволяет применять стандартный стохастический аппарат (diffusion + jumps), не упираясь постоянно в граничные ограничения.

## Локальные чувствительности в logit-координатах

Для логистического отображения $S(x)$:

$$
S'(x)=p(1-p)
$$

$$
S''(x)=p(1-p)(1-2p)
$$

Интерпретация:
- $S'(x)$ задаёт, насколько сильно движение в logit переносится в движение вероятности
- $S''(x)$ описывает кривизну/асимметрию около экстремумов (очень низкие или очень высокие вероятности)

## Слой маркет-мейкинга (logit Avellaneda-Stoikov)

Чтобы котировать в стакане, ядро адаптирует Avellaneda-Stoikov напрямую в logit-единицах.

Входы по каждому рынку:
- $x_t$: текущий logit mid
- $q_t$: текущий инвентарь
- $\sigma_b$: волатильность belief
- $\gamma$: риск-аверсия
- $\tau = T-t$: время до резолва
- $k$: параметр прихода ордеров/ликвидности

### Reservation Quote

Инвентарь сдвигает внутреннюю fair value:

$$
r_x(t)=x_t - q_t\,\gamma\,\overline{\sigma_b^2}\,(T-t)
$$

Чем больше лонг-инвентарь, тем ниже reservation quote (более агрессивная продажа), и наоборот.

### Оптимальный спред (аппроксимация)

Полный спред в logit-единицах:

$$
2\delta_x(t) \approx \gamma\,\overline{\sigma_b^2}\,(T-t) + \frac{2}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

Half-spread, используемый в коде:

$$
\delta_x(t) \approx \frac{1}{2}\gamma\,\overline{\sigma_b^2}\,(T-t) + \frac{1}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

Далее:

$$
x^{bid}=r_x-\delta_x, \quad x^{ask}=r_x+\delta_x
$$

$$
p^{bid}=S(x^{bid}), \quad p^{ask}=S(x^{ask})
$$

## Decision Support & Analytics

Аналитический слой расширяет те же модельные предположения на калибровку, сценарный анализ, sizing, диагностику микроструктуры и портфельную агрегацию рисков.

### 1. Калибровка Implied Belief Volatility

Из наблюдаемых рыночных котировок $(p^{bid}, p^{ask})$ считаем logit-спред:

$$
\Delta_x^{mkt} = \operatorname{logit}(p^{ask}) - \operatorname{logit}(p^{bid})
$$

В модели котирования:

$$
\Delta_x^{model}(\sigma_b) = \gamma\tau\sigma_b^2 + \frac{2}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

Восстанавливаем implied $\sigma_b$ из уравнения:

$$
f(\sigma_b)=\Delta_x^{model}(\sigma_b)-\Delta_x^{mkt}=0
$$

методом Newton-Raphson в векторизованном виде:

$$
\sigma_{n+1}=\max\left(0,\,\sigma_n-\frac{f(\sigma_n)}{f'(\sigma_n)}\right),\quad f'(\sigma)=2\gamma\tau\sigma
$$

Интуиция: инвертируем наблюдаемый спред рынка в латентный уровень belief-волатильности для адаптивного risk control.

`q_t` сохранён в API для единообразия с остальными batch-аналитиками, но в текущую формулу калибровки не входит.

### 2. Vectorized Stress-Testing (What-If)

Для шока $\Delta p$ строим шокированную вероятность и logit-состояние:

$$
p' = \operatorname{clip}(p + \Delta p), \quad x' = \operatorname{logit}(p')
$$

Далее пересчитываем reservation и спред:

$$
r_x' = x' - q_t\gamma\sigma_b^2\tau
$$

$$
\delta_x' = \frac{1}{2}\gamma\sigma_b^2\tau + \frac{1}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

и получаем новые котировки, греки и инвентарный PnL shift:

$$
\Delta \text{PnL} \approx q_t (p' - p)
$$

Интуиция: SIMD what-if карта мгновенно показывает дрейф котировок и риска под вероятностными шоками.

### 3. Adaptive Kelly / Optimal Sizing

Определяем edge относительно рынка:

$$
e = p_{user} - p_{mkt}, \quad v = p_{mkt}(1-p_{mkt})
$$

Kelly-подобный sizing-сигнал:

$$
f^* = \frac{e}{v}
$$

Далее масштабирование и ограничение по риск-бюджету и инвентарю:

$$
\text{clip}_{taker} = \operatorname{clamp}\left(f^* \cdot \frac{\text{risk\_limit}}{1+\gamma|q_t|},\,-\text{max\_clip},\,\text{max\_clip}\right)
$$

с финальными hard-bound по inventory limits; maker clips берутся как более консервативная доля taker clips.

Интуиция: перевод статистического edge в исполнимый размер позиции с учётом инвентарной выпуклости и риск-лимитов.

### 4. Микроструктура стакана (OBI/VWM Pressure)

Дисбаланс top-of-book:

$$
\operatorname{OBI} = \frac{V_b - V_a}{V_b + V_a}
$$

Volume-weighted mid proxy:

$$
\operatorname{VWM} = \frac{p^{ask}V_b + p^{bid}V_a}{V_b + V_a}
$$

Переход к logit и сигнал давления:

$$
\text{pressure} = \operatorname{OBI} + \frac{\operatorname{VWM} - \operatorname{mid}}{\operatorname{spread}}
$$

Интуиция: объединяем очередной дисбаланс и ценовой skew в быстрый направленный фактор микроструктуры.

### 5. Cross-Market Portfolio Greeks

Взвешенные экспозиции по рынкам:

$$
E_i^\Delta = q_i\Delta_i w_i, \quad E_i^\Gamma = q_i\Gamma_i w_i
$$

Без матрицы корреляций:

$$
\Delta_{net}=\sum_i E_i^\Delta, \quad \Gamma_{net}=\sum_i E_i^\Gamma
$$

С матрицей корреляций $C$:

$$
\Delta_{net}=\sum_i E_i^\Delta \sum_j C_{ij}E_j^\Delta
$$

$$
\Gamma_{net}=\sum_i E_i^\Gamma \sum_j C_{ij}E_j^\Gamma
$$

Интуиция: сжимаем кросс-рыночные экспозиции в портфельное risk-state в logit-координатах.

## HFT-детали реализации

Математическая модель реализована с приоритетом на throughput:

- **SoA layout** (`x_t[]`, `q_t[]`, `sigma_b[]`, `gamma[]`, `tau[]`, `k[]`) для contiguous memory stream
- **Portable baseline path**, чтобы крейт запускался на любом x86_64 CPU
- **Runtime-dispatched AVX-512 SIMD** пакетный расчёт по 8 `f64` lane на поддерживаемых хостах
- **Точный контракт sigmoid/logit** в публичном API с численно согласованными portable и AVX-512 путями
- **Ноль аллокаций в hot path** (все буферы передаются вызывающей стороной)

Такой дизайн убирает штрафы `gather` из AoS-layout и при этом сохраняет численную согласованность между portable и AVX-512 путями.

## Численные guard rails

Для устойчивого продакшн-поведения:

- Входы поджимаются там, где нужно (`k >= \epsilon`, `gamma >= 0`, `tau >= 0`)
- На входе `logit` вероятности ограничиваются в $(\epsilon, 1-\epsilon)$, чтобы `logit` оставался конечным
- Math с минимизацией ветвлений помогает держать предсказуемую latency под нагрузкой

## Практическая интуиция

- Термин инвентарного риска расширяет/смещает котировки при росте позиции.
- Волатильность и время до резолва увеличивают risk compensation.
- Параметр прихода ордеров $k$ задаёт нелинейную часть спреда.
- Аналитический слой замыкает контур от наблюдений рынка к исполнимым sizing и портфельному риск-контролю.

Итог: это production-grade, SIMD-native движок котирования, decision-support и управления риском для prediction markets.
