# Tutoriál · 05 · Racing Lab

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Racing Lab je iný než ostatné kroky: je to malý simulátor, ktorému pribudne nová **kapitola** vždy, keď má krok roadmapy koncept, ktorý sa prirodzene hodí na autíčka. Tento tutoriál najprv krok za krokom postaví spoločný simulátor (trať → auto → senzory → vodič → simulačný cyklus → vykresľovanie) a potom vysvetlí, ako ho používa každá kapitola a prečo je postavená takto.

**Cieľ:** svet dosť jednoduchý na to, aby si rozumel každému číslu, ktoré auto vidí (8 vstupov), aj každému číslu, ktoré vracia (2 ovládania), takže každú ML myšlienku môžeš sledovať očami, nie len merať.

**Potrebuješ:** projekty 01–04 (NumPy, matice, autograd). Neskoršie kapitoly používajú PyTorch a scikit-learn.

---

## Časť A · Simulátor ([`racing/`](racing/))

### Krok 1 · Vygeneruj trať ([`racing/track.py`](racing/track.py))

```python
def generate(seed, n_points=360, width=60.0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = np.ones_like(theta)
    for k in (2, 3, 4, 5, 6):
        amp = rng.uniform(0.04, 0.3 / k**0.5)
        r += amp * np.cos(k * theta + rng.uniform(0, 2*np.pi))
    center = [W/2 + rx * r * cos(theta), H/2 + ry * r * sin(theta)]
```

**Ako to funguje:** začni kružnicou (polomer 1) a pridaj pár vĺn: `cos(k·θ)` s k = 2…6 dá 2 až 6 výbežkov okolo dookola, každý s náhodnou veľkosťou a fázou. Súčet je hladká uzavretá krivka, **stredová čiara**.

**Prečo takto:**
- **Vždy uzavretá a hladká**, lebo `cos(kθ)` sa pre celé k opakuje každých 2π.
- **Nepretína sama seba**, lebo amplitúdy sú malé (`< 0,3/√k`) a polomer zostane kladný.
- **Seed = trať.** `TRAIN_SEEDS = 0–9` a `TEST_SEEDS = 100–109` dajú 10 trénovacích tratí a 10 tratí, ktoré vodiči **počas tréningu nikdy nevidia**. Toto rozdelenie train/test umožní neskorším kapitolám pýtať sa „naučil sa jazdiť, alebo si zapamätal 10 tratí?“.

**Steny zo stredovej čiary** (`Track.__post_init__`):
```python
tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)       # smer pozdĺž trate
normal  = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)    # dotyčnica otočená o +90°
self.left  = c + normal * width/2
self.right = c - normal * width/2
```
Otočenie 2-D vektora `(x, y)` o 90° dá `(−y, x)`; to je normála. Posun stredovej čiary doľava a doprava pozdĺž nej dá dve steny, uložené ako krátke úsečky (na vrhanie lúčov).

### Krok 2 · Fyzika áut pre veľa áut naraz ([`racing/car.py`](racing/car.py))

```python
self.speed   = clip(speed + (throttle*ACCEL - DRAG*speed) * DT, 0, MAX_SPEED)
self.heading = heading + steer * TURN_RATE * DT
vel          = [cos(heading), sin(heading)] * speed
self.pos     = pos + vel * DT
```

**Ako to funguje:** arkádová fyzika s tromi stavovými premennými na auto (poloha, smer, rýchlosť) a dvomi ovládaniami v [−1, 1]: `steer` (kladné = doľava) a `throttle` (záporné = brzda). `DT = 1/30 s`, jeden snímok. Odpor (`−DRAG·speed`) dá prirodzenú maximálnu rýchlosť.

**Prečo vektorizovane:** každý atribút je NumPy pole tvaru `(N,)`, takže jeden `step()` pohne všetkými N autami. Kapitola 12 beží so 60 autami; kapitola 06 so 40 na trať. Python cyklus cez autá by bola pomalá cesta z projektu 03.

**Prečo `np.where(live, nové, staré)`:** havarované alebo dojazdené autá musia zamrznúť, kým ostatné idú ďalej, bez toho, aby sa rozbila vektorizácia.

**Postup po trati** (`_update_progress`): každé auto si drží najbližší bod stredovej čiary a hľadá len blízko posledného indexu (`-10 … +30`). Je to lacné a nikdy to „neskočí“ na druhú stranu trate. Postup je súčet prejdených úsekov stredovej čiary (záporný, keď ide dozadu). Keď je auto ďalej než `width/2` od stredovej čiary, vyletelo z cesty: `alive = False`.

### Krok 3 · Senzory: 7 lúčov ako sústavy 2×2

```python
RAY_ANGLES = deg2rad([-90, -60, -30, 0, 30, 60, 90])

def cast_rays(pos, heading, walls):
    d  = ray_directions(heading)           # (N, R, 2) = R(heading + offset) @ [1, 0]
    ...
    det = cross(dd, -e)                    # (N, R, W)
    t   = cross(ap, -e) / det              # vzdialenosť pozdĺž lúča
    u   = cross(dd, ap) / det              # poloha na úsečke steny
    hit = (abs(det) > 1e-9) & (t > 0) & (0 <= u <= 1)
```

**Ako to funguje:** lúč je `p + t·d`, úsečka steny je `a + u·(b − a)`. Stretnú sa tam, kde sú obe rovnaké: dve rovnice (x a y), dve neznáme (t, u). To je lineárna sústava 2×2, vyriešená Cramerovým pravidlom (podiel determinantov, zapísaný ako 2-D vektorové súčiny). Zásah platí, len ak `t > 0` (pred autom) a `0 ≤ u ≤ 1` (na úsečke). `det = 0` znamená, že lúč je so stenou rovnobežný.

**Prečo takto:** je to lineárna algebra z projektu 03 pri skutočnej práci a vďaka broadcastingu sa všetky **autá × lúče × steny** vyriešia naraz, tvar `(N, R, W)`. Minimum `t` cez steny je vzdialenosť k najbližšej stene.

**Pozorovanie** (`Cars.observe`): 7 vzdialeností / 220 a rýchlosť / max → **8 čísel v [0, 1]**. Škálovanie na podobné rozsahy je dôležité: neurónová sieť sa učí zle, keď je jeden vstup 400 a iný 0,1.

### Krok 4 · Ručne napísaný baseline vodič ([`racing/drivers.py`](racing/drivers.py))

```python
left  = rays[:, 4:].mean(1)            # +30°, +60°, +90°
right = rays[:, :3].mean(1)            # −90°, −60°, −30°
steer = steer_gain * (left - right)    # zatoč na stranu, kde je viac miesta
target = clip(rays[:, 3] * aggression * 1.4, 0.25, 1.0)   # rýchlejšie, keď je cesta vpredu dlhá
throttle = clip((target - speed) * 5, -1, 1)
```

**Prečo baseline:** každá kapitola potrebuje niečo na porovnanie a niektoré potrebujú **učiteľa**, ktorého budú kopírovať. Tento vodič na pravidlách dojde 9 z 10 testovacích tratí. Má dva gombíky (`steer_gain`, `aggression`), ktoré kapitola 02 ladí gradient descentom.

**Rozhranie, ktoré sa používa všade:** vodič je čokoľvek volateľné `obs (N, 8) → (steer (N,), throttle (N,))`. Baseline, MLP na micrograde, CNN aj PPO politika ho dodržiavajú, takže sa dajú všetky vložiť do rovnakého simulačného a vyhodnocovacieho kódu.

### Krok 5 · Simulačný cyklus ([`racing/sim.py`](racing/sim.py))

```python
def run(driver, track, n=1, max_time=60.0, on_step=None):
    cars = Cars(track, n)
    for _ in range(int(max_time / DT)):
        if not (cars.alive & ~cars.finished).any(): break
        obs = cars.observe()
        steer, throttle = driver(obs)
        cars.step(steer, throttle)
        if on_step: on_step(cars, obs, steer, throttle)
```

Klasický cyklus **pozoruj → konaj → krok**, ten istý, aký používa reinforcement learning. `on_step` je háčik, aby kapitoly mohli zaznamenávať dáta (napr. snímky senzorov na tréning) bez zmeny cyklu. `lap()` ho obalí pre jedno auto a vráti dojazdil / čas kola / postup.

### Krok 6 · Vykresľovanie ([`racing/render.py`](racing/render.py))

pygame kreslí trať, autá a lúče plus bočný panel s číslami. Ten istý `Canvas` buď otvorí živé okno (`--watch`), alebo bez displeja (`SDL_VIDEODRIVER=dummy`) ukladá snímky pre GIFy v `assets/`. Os y otáča len renderer; fyzika používa matematické súradnice (y hore), takže lineárna algebra zostane čistá.

---

## Časť B · Kapitoly

Každá kapitola je jeden skript v [`chapters/`](chapters/), ktorý zapíše `results_chNN.json`, graf a GIF.

### Kap. 01 · Batching 200 mozgov ([`ch01_batch.py`](chapters/ch01_batch.py))

`MLPBatch` dá každému z N áut vlastné náhodné MLP 8 → 16 → 2 a vyhodnotí ich dvoma spôsobmi:
```python
cyklus:  for i in range(N): h = tanh(obs[i] @ W1[i] + b1[i]) ...
batched: h = tanh((obs[:, None] @ W1)[:, 0] + b1)    # (N,1,8) @ (N,8,16) → (N,1,16)
```
`None` pridá rozmer, takže pozorovanie každého auta je matica 1×8; `@` potom vynásobí všetkých N dvojíc naraz (pravidlo batchovania z projektu 03). Rovnaký výstup (`allclose`), **21× rýchlejšie** pri 200 autách. Ponaučenie z projektu 01, ktoré vidno.

### Kap. 02 · Reťazové pravidlo a gradient descent ([`ch02b_chain_rule.py`](chapters/ch02b_chain_rule.py), [`ch02_gradient.py`](chapters/ch02_gradient.py))

- **Časť A:** auto brzdí silou `b` a musí zastaviť na 250 px. Miesto zastavenia závisí od `b` cez 90 fyzikálnych krokov, takže `dL/db` je reťazové pravidlo s 90 článkami. Vypočítané ručne (spätný cyklus cez čas), numericky aj v PyTorchi: všetky dajú −650,934. Gradient descent nájde `b = 66,5`. Keď auto zastaví, `max(0, ·)` má deriváciu 0 a zablokuje gradient, ako ReLU.
- **Časť B:** loss je priemerný čas kola na trénovacích tratiach ako funkcia `aggression`. Vzorec neexistuje, takže gradient sa odhaduje numericky a GD zníži čas kola 18,2 → 13,1 s. **Prečo klesajúci learning rate:** pri „útese havárií“ loss nie je hladký a pevný krok cez neho stále preskakoval.

### Kap. 03 · Vektory a matice ([`ch03_vectors.py`](chapters/ch03_vectors.py))

Čistá vizualizácia projektu 03: smer = rotačná matica × [1, 0], skalárny súčin so smerom trate = zarovnanie, lúče = lineárne sústavy z Kroku 3.

### Kap. 04 · Imitation learning na mojom micrograde ([`ch04_imitation.py`](chapters/ch04_imitation.py))

1. **Behaviour cloning:** baseline jazdí trénovacie trate; každý 3. snímok sa stane príkladom `(8 senzorov → učiteľov steer, throttle)`. MLP 8 → 8 → 2 postavené z **triedy `Value` z projektu 04** sa trénuje s MSE.
2. **DAgger:** jazdí *študent*, učiteľ označí každý stav, ktorý študent navštívi, nové príklady sa pridajú, pretrénuje sa (3 kolá).

**Prečo DAgger:** učiteľ nikdy nerobí chyby, takže dáta z kopírovania nikdy neukážu „som pri stene, čo teraz?“. Študent zablúdi do stavov, kde učiteľ nikdy nebol, a chyby sa nabaľujú. DAgger zbiera presne tieto stavy. Dojazdené nevidené trate: 5/10 → **7/10**.

### Kap. 06 · Predpoveď zriedkavých havárií ([`ch06_crash_risk.py`](chapters/ch06_crash_risk.py))

Autá s náhodnou agresivitou a zašumeným riadením jazdia; každý snímok má label *„havaruje v nasledujúcich 30 snímkach (1 s)“*. Pozitívnych je len 6,3 % snímok, takže dáta sú prirodzene nevyvážené. Logistická regresia na 8 senzoroch plus pár ďalších príznakoch (`speed/front`, `speed²`, `min(left, right)`) predpovedá riziko.

**Prečo ďalšie príznaky:** logistická regresia kreslí rovnú hranicu; štvorce a podiely ju nechajú trochu sa ohnúť bez prechodu na väčší model.

**Ponaučenie:** ROC-AUC 0,910 vyzerá výborne, ale pri 80 % recall je precision len 0,25: tri falošné poplachy na jeden skutočný. Pointa projektu 06 na reálnych dátach.

### Kap. 11 · Jazda z pixelov ([`ch11_pixels.py`](chapters/ch11_pixels.py))

- **Kamera:** každá trať sa raz rasterizuje (tráva 0, asfalt 0,5, obrubník 1). `camera()` vyberie okno 64×64 **v súradniciach auta** (otočené tak, že „hore“ je vždy dopredu). Egocentrické obrázky znamenajú, že „zatoč doľava“ vyzerá vždy rovnako, nech je auto kdekoľvek.
- **Modely:** CNN a MLP na sploštených pixeloch s podobným počtom váh, trénované s DAgger na MPS. Zrkadlenie obrázkov a otočenie znamienka riadenia zdvojnásobí dáta (všetky trate idú proti smeru hodinových ručičiek, takže pravotočivé zákruty sú inak zriedkavé).
- **Prečo CNN nemá global average pooling:** klasifikátor sa pýta, *či* je niečo na obrázku; vodič potrebuje vedieť, *kde* je cesta. Spriemerovanie cez pozície by to zahodilo, preto hlava sploští.
- **Chyba s Grad-CAM ponechaná ako lekcia:** klasický Grad-CAM predpokladá hlavu s poolingom. Pri sploštenej hlave je správny variant HiResCAM (gradient × aktivácia bunku po bunke).
- **3 seedy na model:** CNN 38,0 vs MLP 34,7 z 50 nevidených tratí v priemere, ale rozsahy sa prekrývajú, preto je to trend, nie dôkaz.

### Kap. 12 · Reinforcement learning s PPO ([`ch12_ppo.py`](chapters/ch12_ppo.py))

Žiadny učiteľ. Auto dostane len **odmenu**: postup od posledného snímku (škálovaný) mínus 1 za haváriu.

- **`Fleet`:** 60 áut na 10 tratiach, ktoré nikdy nezastanú. Havarované auto sa **znovu objaví na náhodnom mieste** (inak by trénovalo len prvú zákrutu). Polovica respawnov je **státím štart**, pridaný po tom, čo sa našli autá, ktoré sa pri rýchlosti 0 otočili a išli opačným smerom.
- **`ActorCritic`:** dve samostatné MLP 8 → 64 → 64. **Actor** vracia stred Gaussovho rozdelenia pre (steer, throttle) plus naučené `log_std`, takže vzorkovanie pridá šum na prieskum. **Critic** predpovedá budúcu odmenu zo stavu. Samostatné siete, aby veľké chyby kritika nerušili actora. Posledná vrstva actora začína malinká, takže každá akcia začína blízko 0.
- **Zber:** 200 krokov s aktuálnou zašumenou politikou; ukladajú sa pozorovania, akcie, ich log-pravdepodobnosti, odmeny a hodnoty kritika.
- **GAE (advantage):**
  ```python
  delta = r[t] + gamma * V[t+1] * alive - V[t]     # prekvapenie v jednom kroku
  running = delta + gamma * lam * alive * running   # vážený súčet budúcich prekvapení
  ```
  Advantage hovorí, *o koľko lepšie táto akcia dopadla, než kritik čakal*. `gamma = 0,99` zľavňuje vzdialenú budúcnosť; `lam = 0,95` vyvažuje skreslenie a šum.
- **PPO update:**
  ```python
  ratio = exp(new_logp - old_logp)
  policy_loss = -min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
  ```
  Dobré akcie sprav pravdepodobnejšími, zlé menej, ale clip odstráni akýkoľvek zisk z posunu politiky o viac než ±20 % od tej, ktorá dáta zozbierala, lebo tieto dáta nehovoria nič spoľahlivé o veľmi odlišnej politike.
- **Hodiny nie sú havária:** keď sa auto zastaví po 40 s, malo ešte budúcnosť, preto sa k jeho poslednej odmene pripočíta `gamma × V(posledný stav)`. Havária ju naozaj ukončí (budúcnosť = 0).

**Výsledok:** dva z troch seedov dojazdia všetkých 50 nevidených tratí a kolo idú o ~34 % rýchlejšie než baseline; jeden seed uviazne v horšom návyku (43/50). Kontrola s kopírovaním presne kopíruje učiteľa, lebo kopírovanie nemôže prekonať svojho učiteľa. **Ponaučenie:** RL môže učiteľa prekonať, ale jeden beh nie je spoľahlivý výsledok; uvádzaj viac seedov.

---

## Spusti

```bash
uv sync
uv run python -m chapters.ch01_batch --watch
uv run python -m chapters.ch04_imitation      # potrebuje 04-autograd-makemore naklonované vedľa tohto repa
uv run python -m chapters.ch12_ppo --watch    # použije uložené váhy
uv run python -m chapters.play                # jazdi sám šípkami
```

## Vyskúšaj sám

1. Pridaj do generátora možnosť tratí v smere hodinových ručičiek a pozri, ako si s nimi poradí politika z kap. 12 (doteraz zatáčala hlavne doľava).
2. Zmeň `RAY_ANGLES` na 5 lúčov. Dojazdí baseline stále 9/10?
3. Nahraj vlastné kolá cez `chapters.play` a použi ich ako učiteľa v kap. 04 (`--demos data/my_laps.npz`).

## Zhrnutie

| Časť | Myšlienka |
|---|---|
| Trať | Kružnica + náhodné harmonické = hladká uzavretá slučka; seedy dajú trénovacie vs nevidené testovacie trate |
| Auto | Vektorizovaná arkádová fyzika; `np.where` zmrazí havarované autá |
| Senzory | Lúč × stena = sústava 2×2, vyriešená pre všetky autá × lúče × steny naraz |
| Vodič | Jedno rozhranie `(N, 8) → steer, throttle` pre pravidlá, MLP, CNN aj RL |
| Kapitoly | Každý koncept roadmapy (batching, reťazové pravidlo, imitácia, nevyváženosť, CNN, RL) ukázaný na tom istom svete |
