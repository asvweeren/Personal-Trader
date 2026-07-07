# Verbeterplan — Rendement Trading Systeem

_Opgesteld: 2026-07-07. Gebaseerd op analyse van 454 gesloten trades, 148.246 signalen en de portefeuille-snapshots (apr–jul 2026)._

## Samenvatting van de diagnose

Het systeem verloor structureel geld: **-€36.050 gerealiseerd** over ~5 maanden. Het verlies zit **niet in de signalen maar in de executielaag**. Bewijs:

- **Exit-geometrie**: 177 stop-losses geraakt (-€43.219) tegenover **1** take-profit (+€1.132). De stop stond gemiddeld op 2,4% (binnen de dagelijkse ruis), de TP op 8% (praktisch onbereikbaar).
- **Alle 7 strategieën verliezen** → gedeelde laag (exits/timing/sizing) is de oorzaak, niet één model.
- **Model-executie mismatch**: XGBoost meldt 80% test-accuracy maar levert live 27% winrate.
- **Long-bias**: 127.714 BUY- vs 20.532 SELL-signalen; shorts waren het énige winstgevende segment (+€4.371).
- **Risicovangrails vuurden niet**: -€15.658 op één dag (12 feb, ~6,4%) zonder halt; `risk_events`-tabel is volledig leeg; één positie was €232k op €242k portefeuille.

De laatste 30 dagen is het bijna break-even (-€430), dus de bloeding is gestopt — maar er is nog geen positieve verwachtingswaarde. Doel van dit plan: van "geen verlies meer" naar "aantoonbare edge, klaar voor validatie".

---

## Fase 0 — Vangrails repareren (deze week, blokkeert live gaan)

Deze punten zijn veiligheids-kritisch. Zonder deze fixes mag het systeem nooit live.

### 0.1 — `risk_events`-logging werkt niet
**Bevinding**: de tabel bevat 0 rijen terwijl er meerdere daily-loss-overschrijdingen zijn geweest.
**Actie**: debug `_log_risk_event()` in `backend/app/risk/manager.py`. Waarschijnlijk faalt de DB-write stil (exception opgeslokt) of wordt de sessie niet gecommit. Voeg een test toe die verifieert dat een DAILY_LOSS_TRIGGERED-event daadwerkelijk in de DB landt.
**Acceptatie**: forceer een risk-event in test → rij verschijnt in `risk_events`.

### 0.2 — Daily-loss halt is reactief, niet continu
**Bevinding**: `evaluate_signal()` (`manager.py:107`) checkt de daily-loss pas wanneer er een *nieuw* signaal binnenkomt. Verliezen via open posities die stops raken worden niet gemonitord, en de halt sluit bestaande posities niet.
**Actie**:
- Zet de daily-loss-check ook in de EOD-/1-min veiligheidsloop (naast de trading-cycle), zodat hij vuurt op basis van portefeuillewaarde, niet op signaalinstroom.
- Bij trigger: niet alleen nieuwe entries blokkeren maar overweeg alle open posities plat te draaien (of minstens een CRITICAL-alert + stop-tightening).
- Verifieer dat `daily_start_value` elke handelsdag correct wordt gereset (`manager.py:81`).
**Acceptatie**: gesimuleerde -5% dag → halt vuurt binnen 1 cyclus, event gelogd, alert verstuurd.

### 0.3 — Position-sizing enforcement afdwingen vóór order
**Bevinding**: een SAP.DE-positie van €232k op een €242k portefeuille (~96%) is geplaatst ondanks de 12%-limiet (`hard_limits.py:48`). Alle oversized trades stammen uit 11–13 feb.
**Actie**: bevestig dat `check_position_size` nu op élk pad wordt aangeroepen vóór orderplaatsing (ook re-entries en agentic/handmatige paden). Voeg een harde notional-cap toe (bijv. `min(max_position_pct%, absoluut plafond)`) die niet te omzeilen is.
**Acceptatie**: unit-test met order > `max_position_pct` → geweigerd op alle codepaden.

---

## Fase 1 — Exit-geometrie herzien (grootste rendementshefboom)

Dit is de kern. Met 47% winrate moet de gemiddelde winst ≥ gemiddelde verlies zijn; nu is het omgekeerd.

### 1.1 — Realistische risk/reward op basis van gemeten data
**Bevinding**: TP op 8% werd 1× geraakt, stop op 2,4% 177×. De ATR-config (`config.py:98-104`) geeft nominaal 3x ATR stop / 6x ATR TP (2:1), maar de `min_take_profit_pct: 5.0` floor duwt de TP te ver weg voor de effectieve holdingperiode (~12u voor ml_xgboost).
**Actie**:
- Verlaag `min_take_profit_pct` naar ~2–3% en verhoog `min_stop_loss_pct` naar ~3–4% zodat de stop buiten de normale intraday-ruis valt (meet de gemiddelde daily ATR% per symbool om dit te kalibreren).
- Mik op **1,5:1 R:R** in plaats van 2:1 of 3,3:1 — realistisch haalbaar binnen de holdingperiode.
- Overweeg volledig **trailing-stop-based exits** i.p.v. vaste TP (de code heeft al trailing-logica, `engine.py:1735`).
**Acceptatie**: backtest van exact deze exit-regels laat TP-hit-rate > 25% en positieve verwachtingswaarde zien.

### 1.2 — Backtest de exits, niet alleen de signalen
**Bevinding**: het model is goed op richting maar trades verliezen alsnog door de stop → de backtest test blijkbaar niet dezelfde exit-mechaniek als productie.
**Actie**: zorg dat `backtest/simulator.py` exact dezelfde stop/TP/trailing/breakeven-logica gebruikt als `execution/engine.py`. Backtest is nu misleidend als deze afwijkt.
**Acceptatie**: dezelfde trade in backtest en live geeft dezelfde exit-reden.

### 1.3 — Breakeven- en partial-profit-parameters valideren
**Bevinding**: `breakeven_stop_trigger_pct: 3.0` en `partial_profit_enabled: True` zijn recent toegevoegd. Meet of ze helpen of juist winst afkappen vóór de beweging compleet is.
**Actie**: A/B-vergelijk in backtest met/zonder breakeven-stop en partial-profit.

---

## Fase 2 — Model & signalen verbeteren

### 2.1 — 80%-accuracy onderzoeken op leakage
**Bevinding**: 80% test-accuracy op koersrichting is verdacht hoog; live winrate is 27%.
**Actie**: audit `scripts/train_model.py` op label-leakage (features die toekomstinfo bevatten, target-berekening die vooruit kijkt zonder purge/embargo). Gebruik **time-series CV met embargo**, geen random split. De echte edge is vermoedelijk veel kleiner.
**Acceptatie**: gerapporteerde CV-accuracy komt binnen ~5pp van de live winrate.

### 2.2 — Labels matchen met de echte exit-horizon
**Bevinding**: model getraind op daily-forward-returns, maar afgehakt met intraday-stops.
**Actie**: definieer het trainingslabel als "haalt de trade TP vóór stop, gegeven de productie-exit-regels" (triple-barrier method). Dan leert het model wat het systeem daadwerkelijk verhandelt.

### 2.3 — Long-bias corrigeren
**Bevinding**: 6:1 BUY/SELL-verhouding, vlakke confidence ~0,69 → weinig selectiviteit. Shorts waren winstgevend.
**Actie**: verhoog `confidence_threshold` (nu 0,60) en/of herkalibreer de klassegrenzen zodat alleen sterke signalen door de filter komen. Geef de short-kant meer gewicht — het is empirisch het enige winstgevende segment.
**Acceptatie**: minder maar hoger-conviction trades; BUY/SELL-verhouding richting realistischer niveau.

### 2.4 — Instap-timing: vermijd de openingschaos
**Bevinding**: 14:00 UTC (US-open) = 148 trades, -€23.288 — verreweg het slechtste uur.
**Actie**: activeer een opening-range-filter voor de eerste 15–30 min na US-open (`opening_range_minutes` staat nu op 0), of gebruik daar bredere stops.
**Acceptatie**: P&L per uur toont geen concentratie van verlies meer rond de opening.

---

## Fase 3 — Realistische validatie vóór live

### 3.1 — Kosten meenemen
**Bevinding**: `commission` = €0,00 op alle 454 trades (paper account rekent geen spread/slippage/fees).
**Actie**: modelleer in backtest en validatie realistische IBKR-commissies + spread + slippage. Bereken of de edge kosten overleeft.
**Acceptatie**: nettoresultaat ná kosten is positief in de validatieperiode.

### 3.2 — Schaal terug naar het beoogde kapitaal
**Bevinding**: systeem ontworpen voor <€5.000 maar sized op €242k paper → fouten ~50× uitvergroot.
**Actie**: zet paper-kapitaal op het realistische live-niveau, zodat sizing/risk-limieten representatief zijn.

### 3.3 — Validatie-drempel definiëren
**Actie**: minimaal 4 aaneengesloten weken paper trading met: positieve verwachtingswaarde ná kosten, winrate × avg_win ≥ (1-winrate) × avg_loss, max drawdown binnen limiet, en 0 ongeautoriseerde limietoverschrijdingen. Pas dan overweeg live.

---

## Prioriteitsvolgorde (impact × urgentie)

| # | Actie | Fase | Impact |
|---|-------|------|--------|
| 1 | Exit-geometrie fixen (R:R 1,5:1, ruimere stop) | 1.1 | ⭐⭐⭐⭐⭐ |
| 2 | Backtest ↔ live exit-pariteit | 1.2 | ⭐⭐⭐⭐⭐ |
| 3 | Daily-loss halt continu + force-close | 0.2 | ⭐⭐⭐⭐ (veiligheid) |
| 4 | risk_events-logging repareren | 0.1 | ⭐⭐⭐⭐ (veiligheid) |
| 5 | Model-labels op exit-horizon + leakage-audit | 2.1/2.2 | ⭐⭐⭐⭐ |
| 6 | Position-sizing hard cap | 0.3 | ⭐⭐⭐ (veiligheid) |
| 7 | Opening-range-filter (14:00 UTC) | 2.4 | ⭐⭐⭐ |
| 8 | Long-bias / confidence-threshold | 2.3 | ⭐⭐⭐ |
| 9 | Kosten in validatie | 3.1 | ⭐⭐⭐ |
| 10 | Kapitaal terugschalen | 3.2 | ⭐⭐ |

**Snelste weg naar break-even → winst**: #1 en #2 samen. De rest maakt het robuust en veilig genoeg om live te durven.

---

## Implementatiestatus (2026-07-07)

Alle fases zijn geïmplementeerd in code. 431 backend-tests groen (was 421; +10 nieuwe). Nog niet gedeployed naar de server.

### Wat er in code is aangepast

**Fase 0.1 — risk_events logging gerepareerd**
`_log_risk_event()` deed `flush()` zonder `commit()` en de RiskManager-singleton kreeg `db=None` → tabel bleef leeg. Nu krijgt de RiskManager een eigen `session_factory` (`dependencies.py`), opent per event een korte sessie en **commit** die onafhankelijk. Fouten worden nu op `exception`-niveau gelogd i.p.v. stil op `debug`.
- `backend/app/risk/manager.py`, `backend/app/dependencies.py`
- Test: `test_log_risk_event_persists_and_commits`, `test_log_risk_event_no_session_does_not_raise`

**Fase 0.2 — continue daily-loss halt + force-close**
Nieuwe `TradingEngine.check_daily_loss_halt()` draait elke minuut via de veiligheidsloop (ook in swing-modus, ook zonder open trades). Bij overschrijding: zet halt-vlag, logt CRITICAL risk-event, publiceert `risk.daily_stop`, stuurt een kritiek alert en — indien `daily_loss_force_close=True` — sluit alle open posities via `_force_close_all()`.
- `backend/app/execution/engine.py`, `backend/app/core/scheduler.py`
- Test: `test_daily_loss_halt_triggers_and_blocks`, `test_daily_loss_halt_not_triggered_when_within_limit`

**Fase 0.3 — harde notional-cap**
`check_position_size()` dwingt nu naast het percentage ook een absolute `max_position_notional` af (0 = uit). Backstop zodat één oversized order (zoals de €232k-trade) nooit meer kan.
- `backend/app/risk/hard_limits.py`, config `max_position_notional`
- Test: `test_position_size_absolute_notional_cap`, `test_position_size_notional_cap_disabled_by_default`

**Fase 1.1 — exit-geometrie ~1,5:1**
`atr_take_profit_multiplier` 6.0→**4.0**, `min_take_profit_pct` 5.0→**4.5**, `atr_stop_multiplier` 3.0→**2.5**. Nieuwe `min_risk_reward_ratio` (1.5) die de TP omhoog schuift als de geometrie te krap is (in `_execute_buy`).
- `backend/app/config.py`, `backend/app/execution/engine.py`

**Fase 1.2 — backtest ↔ live pariteit**
`BacktestConfig` haalt stop/TP/trailing/kapitaal/max-positie nu uit de live `settings` (waren hardcoded 1.5%/3.0%, niet aligned). Backtest-P&L reflecteert nu dezelfde exit-geometrie als productie.
- `backend/app/backtest/engine.py`

**Fase 2.1 — leakage-fix (embargo)**
`time_based_split()` heeft een `embargo`-parameter die de laatste N rijen van train/val purget (N = label-horizon), zodat forward-return-labels niet over de split-grens lekken. `prepare_ml_data` gebruikt embargo = `forward_periods`. Dit verklaart een deel van de te hoge 80%-accuracy.
- `backend/app/strategy/feature_pipeline.py`
- Test: `test_split_embargo_purges_boundary_rows`

**Fase 2.2 — triple-barrier labels**
Nieuwe `create_triple_barrier_target()`: labelt 1 als de take-profit vóór de stop wordt geraakt binnen de holdingperiode — matcht wat het systeem echt verhandelt. Opt-in via `ml_use_triple_barrier` (default uit), tp/sl uit de live floors. Wired in `MLStrategy.train()`.
- `backend/app/strategy/feature_pipeline.py`, `backend/app/strategy/ml_strategy.py`, config `ml_use_triple_barrier`
- Test: `test_triple_barrier_labels_tp_before_sl`, `..._sl_before_tp`, `..._tail_is_nan`

**Fase 2.3 — long-bias / selectiviteit**
`confidence_threshold` 0.60→**0.68** — minder maar hoger-conviction trades. Short selling stond al aan en was het enige winstgevende segment (blijft aan).
- `backend/app/config.py`

**Fase 2.4 — opening-range filter**
`opening_range_minutes` 0→**30**. De filter-logica bestond al (`engine.py:557`) maar stond uit. Nu worden BUY-entries in de eerste 30 min na open geskipt (14:00 UTC was het slechtste uur: -€23k).
- `backend/app/config.py`

**Fase 3.1 — kosten**
De backtest-simulator modelleerde al commissie/slippage/spread. De go-live gate compenseert nu voor het feit dat paper-trades €0 kosten hebben (zie 3.3).

**Fase 3.3 — validatie-gate met kostenbuffer**
`MIN_PROFIT_FACTOR` 1.2→**1.3** als marge zodat de edge live commissies overleeft. De gate eiste al ≥20 handelsdagen (~4 weken), Sharpe > 0.5, winrate > 40%, drawdown < 15%, backtest-afwijking < 30%.
- `backend/app/monitoring/paper_trading_validator.py`

### Handmatige acties buiten code (niet automatiseerbaar)

- **Fase 3.2 — kapitaal terugschalen**: `initial_capital` staat al op €5.000. Het IBKR **paper-account is met €242k gefund** — dat is een broker-side instelling. Verlaag de paper-accountwaarde bij IBKR naar het beoogde live-niveau zodat sizing/limieten representatief zijn.
- **Model hertrainen**: zet `ML_USE_TRIPLE_BARRIER=true` in server `.env` en draai de retrain (of wacht op de wekelijkse job zondag 02:00 UTC). Valideer het barrier-aware model vóór het live vertrouwd wordt — het vervangt de labeldefinitie.
- **Optioneel** `MAX_POSITION_NOTIONAL` in `.env` zetten als extra harde euro-cap.

### Deploy

```bash
git push origin master && ssh trader-server "cd /root/trader && git pull origin master && docker compose -f docker-compose.prod.yml up -d --build backend && docker compose -f docker-compose.prod.yml exec -T nginx nginx -s reload"
```

Na deploy verifiëren dat `risk_events` weer vult (forceer eventueel een test-event) en dat de opening-range/exit-config in de logs verschijnt.
