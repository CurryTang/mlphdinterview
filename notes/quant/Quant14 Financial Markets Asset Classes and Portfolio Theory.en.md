# Quant 10 · Markets, assets and portfolios

Course: [[Quant13 Game Theory and Strategic Decision Making|09 Game Theory]] → This note → [[Quant16 Linear Regression Kernel Smoothing and Interview Classics|11 Regression]]

Understanding the mechanics of the financial system is a prerequisite for strategy implementation. The core of the financial system lies in intertemporal value exchange and risk reallocation. Asset pricing models, derivative analysis, and quantitative trading systems all operate on this foundational architecture.

---

## 1 · Market architecture and liquidity ecosystem

### Primary and secondary markets

The lifecycle of financial assets begins with capital formation and relies on continuous turnover for price discovery.

```text
Primary market
Core function: Capital formation and real-economy financing
Participant behavior: Corporations and governments issue new securities; funds flow directly to issuers
Typical scenarios: IPO, private placement, bond issuance

Secondary market
Core function: Stock circulation and price discovery
Participant behavior: Investors trade among themselves; issuers do not participate directly
Typical scenarios: Stock trading, derivative position closing
```

The secondary market provides an immediate liquidation channel for investors. Liquidity premium is an important component of asset valuation.

### Organizational forms of trading venues

Markets are classified into three types based on standardization and clearing mechanisms:

| Organizational form | Core features | Typical underlying assets |
|---|---|---|
| Exchange | Highly standardized contracts (uniform face value and maturity). Features a central counterparty (CCP) and daily mark-to-market settlement, eliminating bilateral default risk. Information is broadcasted via a central limit order book (CLOB). | Spot stocks, futures, listed options |
| Over-the-counter (OTC) | Bilateral negotiation between buyers and sellers, with fully customized terms. Counterparty credit risk exists. Usually governed by ISDA agreements. | Interest rate swaps (IRS), credit default swaps (CDS), FX forwards |
| Dark pool | Order sizes and expected prices are not publicly disclosed. Used for executing large block trades confidentially to reduce market impact costs. | Institutional block equity trades |

### Limit order book (CLOB) matching mechanism

In a central limit order book, all unexecuted resting orders are sorted by "price-time priority".
- Bids are arranged from highest to lowest price; asks are arranged from lowest to highest price.

- The difference between the highest bid and lowest ask is the bid-ask spread.

- Market orders immediately match at the best available price, extracting liquidity and incurring slippage costs.

- Limit orders rest in the order book to provide liquidity, waiting to be consumed by other market orders.

Market makers continuously post limit orders to provide liquidity, while programmatic traders and institutional hedging algorithms use market orders to rapidly adjust positions.

### Market participants breakdown

Liquidity and price formation in financial markets are handled by three types of institutions:

1. **Buy-side**
   - Holds capital and decision-making authority.
   - Includes asset owners providing long-term capital (pension funds, sovereign wealth funds), traditional asset managers pursuing relative benchmark returns, and hedge funds pursuing absolute returns.
   - Hedge funds use leverage and long/short instruments to isolate systematic risk and capture independent risk premiums.

2. **Sell-side**
   - Provides execution channels and intermediary services.
   - Investment banking divisions assist in primary market pricing and distribution.
   - Prime brokerage businesses provide margin leverage, stock borrowing pools, clearing and custody, and direct market access (DMA).

3. **Market maker**
   - Core model: Does not engage in directional speculation. Earns the bid-ask spread by simultaneously quoting bids and asks on both sides of the order book.
   - Core risks: Adverse selection (pricing lag when facing informed traders) and inventory risk (unbalanced one-sided position exposure).
   - Market making algorithms dynamically adjust bilateral quotes at the microsecond level and transfer inventory exposure using highly correlated derivative instruments.

---

## 2 · Spot assets: equities, fixed income and commodities

Across asset classes, different assets reflect different macroeconomic dimensions and cash flow characteristics.

### Equities and short selling mechanisms

Common stocks represent a residual claim on corporate assets and future free cash flows.

In a bankruptcy liquidation scenario, the payout order in the capital structure is strictly as follows:

```text
Senior secured debt
-> Senior unsecured debt
-> Subordinated debt
-> Preferred stock
-> Common stock
```

Common equity holders sit at the end of the payout chain, bearing the highest risk, and thus earn an equity risk premium over the long term.

Short selling is the foundation of quantitative long/short strategies.
- Borrowing cost: Before shorting, underlying stocks must be borrowed from a prime broker. Highly liquid blue-chip stocks have extremely low borrowing rates; small-cap or heavily shorted stocks become hard-to-borrow, with annualized borrowing costs potentially surging to extreme levels.

- Short squeeze: If the underlying stock price rises sharply, the maintenance margin of the short account will be breached. Forced liquidation by brokers translates into passive market buy orders, causing the price to skyrocket in an exponential positive feedback loop.

### Bonds and fixed income

Bond interest rates form the center of gravity for financial asset pricing. The fair value of any asset is the discounted sum of its expected future cash flows:

$$P = \sum_{t=1}^{T} \frac{CF_t}{(1 + r)^t}$$

An increase in the discount rate $r$ drastically shrinks the present value of forward cash flows.

- **Yield curve**: A curve connecting the yields of government bonds with different maturities. Normally upward-sloping, reflecting the term premium. If short-term yields exceed long-term yields, an inversion occurs, often viewed as a leading indicator of monetary tightening and cyclical recession.

- **Credit spread**: The yield of a corporate bond minus the yield of a risk-free government bond of the same maturity. The spread prices the default probability of the issuer. During liquidity contractions, credit spreads widen sharply.

- **Duration**: Measures the linear sensitivity of bond prices to parallel shifts in interest rates. The longer the duration, the more sensitive the asset is to discount rate changes. Long-duration assets face immense valuation pressure during rate-hiking cycles.

- **Convexity**: The second derivative of the price-yield relationship. When rates fall, bond prices accelerate upward; when rates rise, prices decelerate downward. Convexity provides a nonlinear cushion for bond longs.

### ETF and physical arbitrage mechanisms

ETFs maintain a tight peg between market price and underlying net asset value (NAV) through primary market creation/redemption mechanisms. Authorized Participants (APs) perform the cross-market arbitrage function.

| Market state | AP arbitrage path | Market price feedback |
|---|---|---|
| Secondary market premium | Buy underlying constituent stocks in the equity market, physically create ETF shares with the fund company, and sell the ETF in the secondary market. | ETF supply increases, pushing the market price down toward NAV. |
| Secondary market discount | Buy ETF shares at a low price in the secondary market, redeem them with the fund company for underlying constituent stocks, and sell them in the equity market. | ETF supply decreases, pulling the market price up toward NAV. |

### Foreign exchange and commodities

Foreign exchange is quoted continuously 24 hours a day by the global interbank network, making it the most liquid asset class.
- Carry trade: Borrowing low-interest currencies to buy high-yielding currency assets to earn the interest rate differential. This strategy is prone to liquidation stampedes during macroeconomic turbulence, causing the funding currency to appreciate rapidly and wiping out accumulated interest gains.

Commodity pricing is constrained by spot supply-demand dynamics and storage-transportation costs.
- Contango: Forward prices are higher than near-term prices. Usually stems from spot oversupply; buyers must compensate sellers for storage and insurance costs.

- Backwardation: Near-term prices are higher than forward prices. Stems from extreme spot shortages; physical industries are willing to pay a massive premium (convenience yield) to secure spot inventory.

---

## 3 · Automated market maker (AMM) basics

Decentralized finance, constrained by throughput and computational costs, typically employs liquidity pools and automated functions instead of order books.

### Constant product market maker (CPMM)

The liquidity pool maintains a constant product of the reserves of two tokens:

$$x \cdot y = k$$

When a trader attempts to swap $\Delta x$ amount of token $X$ for token $Y$, the pool must satisfy the following post-trade condition:

$$(x + \Delta x)(y - \Delta y) = k$$

Solving for the amount of $Y$ received:

$$\Delta y = \frac{y \Delta x}{x + \Delta x}$$

Under this curve, the larger the single trade size $\Delta x$ relative to the pool depth $x$, the more severe the marginal execution price deviation. This mechanism naturally creates nonlinear price slippage.

### Impermanent loss

Liquidity providers (LPs) bear the structural cost of passive asset ratio adjustments. When external market prices change, arbitrageurs withdraw the appreciating asset from the pool, leaving behind the depreciating asset.
Assuming the initial prices of two assets are equal, and relative prices subsequently change. Compared to statically holding the assets from inception, the LP's capital change rate can be derived as:

$$\text{IL}(k) = \frac{2\sqrt{k}}{1+k} - 1 \le 0$$

Here, $k$ is the multiplier of the external price change. Whether the price deviates upwards or downwards, $\text{IL}$ is always negative. This payoff profile is equivalent to shorting a straddle in traditional finance: earning transaction fees while bearing the negative gamma convexity loss of large price swings in the underlying asset.

---

## 4 · Derivative pricing and hedging systems

The core economic function of derivatives is to strip risk from assets, enabling precise risk transfer between hedgers and risk bearers.

### Forward and futures pricing

| Dimension | Forwards | Futures |
|---|---|---|
| Trading and customization | OTC trading, highly customizable maturity and size | Exchange-listed, strictly standardized specifications |
| Credit risk | Bilateral default risk exists | Clearinghouse acts as central counterparty, no counterparty credit risk |
| Settlement process | One-time cash or physical settlement at maturity | Initial margin required, daily mark-to-market settlement of floating PnL |

Futures prices are determined by the cost of carry model.
Assume the spot price is $S_0$, the risk-free rate is $r$, the underlying asset has a continuous dividend yield $q$, and the time to delivery is $T$. To preclude risk-free arbitrage, the theoretical futures price must satisfy:

$$F = S_0 e^{(r - q)T}$$

If the listed futures price is higher than this theoretical value, quantitative arbitrageurs will execute cash-and-carry arbitrage: borrow funds to buy the spot asset while simultaneously selling the futures contract. At maturity, they deliver the spot asset to repay principal and interest, locking in a risk-free spread. This arbitrage force rapidly compresses the market price back to theoretical levels.

### Swaps

Swap contracts allow equal institutions to exchange cash flows with different attributes, and their notional principal size is massive.

- **Interest rate swap (IRS)**: Two parties exchange fixed-rate and floating-rate cash flows on the same notional principal. This is widely used for duration management and interest rate risk hedging on balance sheets, with no principal exchange occurring at any point.

- **Credit default swap (CDS)**: A derivative insurance contract against the default risk of a specific debt issuer. The buyer makes periodic premium (spread) payments; upon a default event, the seller must compensate for the bond loss at par value. During the 2008 financial crisis, because institutions without underlying debt exposure were allowed to buy CDS speculatively, the payout chains triggered by defaults directly caused systemic liquidity exhaustion.

### Options and risk asymmetry

Options deconstruct linear rights and obligations. Buyers pay a premium to obtain the right to exercise, with maximum losses strictly capped; sellers collect the premium, bearing unconditional performance obligations and directional exposure.

Anatomy of option value:

$$\text{Option price} = \text{Intrinsic value} + \text{Time value}$$

- Intrinsic value: The theoretical profit that could be realized if the option were exercised immediately.

- Time value: The portion of the option price exceeding intrinsic value, reflecting the expectation that further asset volatility before maturity will yield a more favorable outcome. When the spot price exactly equals the strike price (at-the-money), uncertainty is highest, and time value reaches its absolute peak.

**Put-call parity**
For European options on the same underlying asset with the identical strike price $K$ and time to maturity $T$, the static replication equation must strictly hold:

$$C - P = S - K e^{-rT}$$

Buying a call and selling a put creates a payoff perfectly equivalent to holding the spot asset and borrowing the present value of the strike. If market quotes deviate from this equation, market makers will immediately execute risk-free conversion or reversal arbitrage.

**Early exercise logic**
For American call options on non-dividend-paying stocks, early exercise should never occur. Early exercise only captures intrinsic value, whereas closing the position in the secondary market recovers the remaining time value, and delaying the strike payment saves interest costs.

---

## 5 · BSM model and the Greeks

The Black-Scholes-Merton framework demonstrates that option pricing is independent of subjective directional forecasts, transforming the pricing process into an objective replication cost calculation.

### Dynamic delta replication

By holding an option and executing reverse hedging with the underlying spot asset, a portfolio can be immunized against minute fluctuations in the underlying asset over infinitesimally small time intervals. When a portfolio achieves completely non-directional risk, its expected return must equal the risk-free rate.
The BSM partial differential equation derived from this serves as the analytical benchmark for derivative valuation. Key inputs in the pricing model include the spot price, strike price, risk-free rate, time to maturity, and expected volatility.

### Implied volatility

Since future realized volatility is unobservable, traders plug the actual traded market price of the option backwards into the BSM formula; the resulting volatility is called implied volatility (IV). It represents the market's consensus pricing of future asset volatility.

- **Volatility skew**: In equity index options, the implied volatility of out-of-the-money puts is significantly higher than that of at-the-money options. This reflects market participants' structural hedging demand against systemic tail risk (crashes), as well as the leverage effect where a stock price decline mechanically increases the firm's financial leverage.

### The Greeks risk matrix

Quantitative market making and portfolio management use the Greeks to measure non-linear exposures.

| Risk dimension | Mathematical definition | Trading interpretation |
|---|---|---|
| **Delta ($\Delta$)** | $\frac{\partial V}{\partial S}$ | The linear sensitivity of option price to changes in the underlying asset price. Represents the equivalent spot position. |
| **Gamma ($\Gamma$)** | $\frac{\partial^2 V}{\partial S^2}$ | The sensitivity (curvature) of Delta to underlying asset price changes. At-the-money options have the largest Gamma. |
| **Theta ($\Theta$)** | $\frac{\partial V}{\partial t}$ | The decay rate of option value as time passes. Longs pay Theta, shorts collect Theta. |
| **Vega ($
u$)** | $\frac{\partial V}{\partial \sigma}$ | The sensitivity of option price to changes in implied volatility. |
| **Rho ($ho$)** | $\frac{\partial V}{\partial r}$ | The sensitivity of option price to changes in the risk-free interest rate. |

Being long Gamma (holding long options) allows one to profit from violent swings in the underlying asset, but this must be paid for via daily Theta decay. This embodies the conservation of energy law in option pricing:
$$\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma \approx 0$$
If the actual realized volatility falls short of the priced implied volatility, the long position will face systemic net losses.

---

## 6 · Portfolios and return decomposition

Modern portfolio theory utilizes incomplete correlation between assets to optimize the risk-return ratio.

### Diversification and the Capital Asset Pricing Model

Combining multiple assets with correlation coefficients $ho < 1$ can reduce total portfolio variance without lowering expected returns.
The Capital Asset Pricing Model (CAPM) decomposes the excess return of a portfolio into two orthogonal components:

$$R_p = R_f + eta (R_m - R_f) + lpha$$

- **Beta ($eta$)**: The asset's systematic risk exposure relative to broad market fluctuations.

- **Alpha ($lpha$)**: The specific, idiosyncratic excess return generated by the asset itself, independent of systematic risk. Stripping away Beta to capture pure, robust Alpha is the core objective of quantitative strategies.

### Classic asset allocation schools and risk contribution

The industry has evolved several representative asset allocation philosophies:

| School | Asset allocation pattern | Underlying logic |
|---|---|---|
| Classic 60/40 portfolio | 60% equities + 40% bonds | Equities capture economic growth returns, while bonds provide a defensive cushion during economic downturns and recessions. |
| Risk parity | Introduces leverage to dynamically allocate equities, bonds, and commodities | Abandons fixed dollar allocation, ensuring each asset class contributes strictly equally to the total portfolio variance. |
| Endowment model | Heavy allocation to private equity, real estate, and absolute return instruments | Sacrifices short-term liquidity to capture the excess risk premium of long-term illiquid assets. |

In a traditional 60/40 portfolio, because equity volatility (approx. 16%) is far higher than bond volatility (approx. 5%), over 90% of the portfolio's variance is actually contributed entirely by the equity side. Capital allocation percentages absolutely do not equate to actual risk bearing percentages.

---

## 7 · Quantitative financial analysis tools

The following Python scripts are used for option pricing analysis and portfolio risk decomposition.

### Option analytical pricer and Greeks calculation

Computes European option prices and core sensitivity metrics using the BSM analytical solution.

```python
import math
from typing import Dict, Literal

class BSMAnalytics:
    """Option pricing and Greeks analysis tool"""

    @staticmethod
    def _phi(x: float) -> float:
        """Standard normal probability density function"""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _cdf(x: float) -> float:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def price_and_greeks(
        cls,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: Literal["call", "put"] = "call",
    ) -> Dict[str, float]:
        """Calculates theoretical option price and various Greeks"""
        s, k, t, r, sigma = spot, strike, time_to_maturity, risk_free_rate, volatility

        if t <= 0:
            payoff = max(s - k, 0.0) if option_type == "call" else max(k - s, 0.0)
            return {"price": payoff, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        sqrt_t = math.sqrt(t)
        d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        n_d1 = cls._phi(d1)
        cdf_d1 = cls._cdf(d1)
        cdf_d2 = cls._cdf(d2)

        df = math.exp(-r * t)

        if option_type == "call":
            price = s * cdf_d1 - k * df * cdf_d2
            delta = cdf_d1
            theta = -(s * n_d1 * sigma) / (2.0 * sqrt_t) - r * k * df * cdf_d2
        else:
            price = k * df * cls._cdf(-d2) - s * cls._cdf(-d1)
            delta = cdf_d1 - 1.0
            theta = -(s * n_d1 * sigma) / (2.0 * sqrt_t) + r * k * df * cls._cdf(-d2)

        gamma = n_d1 / (s * sigma * sqrt_t)
        vega = s * sqrt_t * n_d1

        return {
            "price": round(price, 4),
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta_daily": round(theta / 365.0, 4),
            "vega_1pct": round(vega / 100.0, 4)
        }

if __name__ == "__main__":
    res = BSMAnalytics.price_and_greeks(
        spot=100.0, strike=100.0, time_to_maturity=1.0, 
        risk_free_rate=0.05, volatility=0.20, option_type="call"
    )
    for key, value in res.items():
        print(f"  {key:18s}: {value}")
```

### Portfolio true risk contribution measurement

Decomposes the marginal risk contribution of major asset classes to total portfolio volatility based on Euler's theorem.

```python
def portfolio_risk_breakdown(
    weight_stock: float,
    weight_bond: float,
    vol_stock: float = 0.18,
    vol_bond: float = 0.06,
    corr: float = 0.0,
):
    """Calculates total portfolio variance and true risk contribution percentage of assets"""
    w_s, w_b = weight_stock, weight_bond
    sigma_s, sigma_b = vol_stock, vol_bond

    # Total portfolio variance and volatility
    variance_total = (
        (w_s * sigma_s) ** 2
        + (w_b * sigma_b) ** 2
        + 2 * w_s * w_b * sigma_s * sigma_b * corr
    )
    vol_total = variance_total**0.5

    # Marginal risk contribution
    mrc_stock = (w_s * sigma_s**2 + w_b * sigma_s * sigma_b * corr) / vol_total
    mrc_bond = (w_b * sigma_b**2 + w_s * sigma_s * sigma_b * corr) / vol_total

    # Total risk contribution
    trc_stock = w_s * mrc_stock
    trc_bond = w_b * mrc_bond

    pct_stock = (trc_stock / vol_total) * 100
    pct_bond = (trc_bond / vol_total) * 100

    print(f"Capital allocation: Stocks {w_s*100:.0f}% / Bonds {w_b*100:.0f}%")
    print(f"Annualized portfolio volatility: {vol_total*100:.2f}%")
    print(f"Actual risk bearing -> Stocks: {pct_stock:.2f}% | Bonds: {pct_bond:.2f}%\n")


if __name__ == "__main__":
    print("Classic 60/40 portfolio analysis:")
    portfolio_risk_breakdown(weight_stock=0.60, weight_bond=0.40)
    
    print("Risk parity portfolio analysis:")
    portfolio_risk_breakdown(weight_stock=0.25, weight_bond=0.75)
```

---

## References

- Hull, J. C. (2014). *Options, futures, and other derivatives*. Pearson.
