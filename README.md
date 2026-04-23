# Tech-Stock-Market-Analysis
The notebook is doing four jobs in sequence: it prepares stock data, engineers indicators, visualizes market behavior from multiple angles, and then tests simple ML models. The point is not just to “make charts,” but to answer different questions about performance, risk, momentum, relationships between stocks, and predictability.


1. Setup and configuration

The first block imports libraries like pandas, numpy, matplotlib, seaborn, scipy, and sklearn. Each has a clear role:

pandas handles stock tables, dates, grouping, and aggregation.
numpy handles vectorized math like returns, volatility, and rolling computations.
matplotlib and seaborn create the charts.
scipy.stats computes skewness and kurtosis for return distributions.
sklearn trains regression models and evaluates prediction quality.
The color dictionary gives each stock a consistent identity throughout the notebook. That matters because the same colors are reused in all figures, making visual comparisons easier.

The dark_style() function sets a custom plotting theme. This is purely presentation, but helpful when there are many dense charts. It improves readability and consistency.

Why we need this section:

It ensures every later cell runs without re-declaring imports.
It creates a clean visual language for the full analysis.
It avoids repeated formatting logic in every chart block.
2. Load and clean data

In the load section, the script reads one CSV per ticker, standardizes column names, converts Date to datetime, sorts rows chronologically, and combines all stocks into one dataframe.

Then it creates time-based columns:

Year
Quarter
Month
DayOfWeek
These are useful because later figures summarize data by year, quarter, and month.

It also creates derived market features:

Daily_Return: percentage change in close price
Price_Range: daily high minus daily low
Volatility: absolute daily return
Up_Down: whether the daily return is positive or negative
Log_Return: log-based return between consecutive closes
Why we need this:

Raw OHLCV data alone is not enough for analysis.
Most finance insights come from transformed values like returns, volatility, trend, and momentum.
These columns are reused across almost every figure and the ML section.
Conclusion of this section:

You transform raw price history into analysis-ready market behavior data.
3. Technical indicators

The add_indicators() function adds classic technical analysis features for each stock independently:

MA_20, MA_50, MA_200: moving averages for short-, medium-, and long-term trend
EMA_12, EMA_26: exponential moving averages
MACD: momentum indicator based on EMA difference
Signal: smoothed MACD line
RSI: relative strength index, often used to detect overbought/oversold conditions
BB_Upper, BB_Lower, BB_Width: Bollinger Band boundaries and width
Norm_Close: normalized close price starting from 100
Why we need this:

These features turn plain price data into interpretable indicators.
They help answer trend, momentum, and volatility questions.
They also feed the ML models later.
Conclusion:

This is the “feature engineering” core of the notebook.
Figure 1: Normalized Price Performance, RSI, and Volume
See around stock_analysis.py

What it shows:

Top panel: normalized price paths so all stocks start at the same baseline
Middle panel: RSI over time
Bottom panel: daily total trading volume
Use case:

Compare relative growth fairly, even when stocks have different price levels.
Spot overbought/oversold zones with RSI.
Check whether large moves happen alongside heavy trading activity.
Why we need it:

Absolute prices are misleading for comparison.
A $50 move in one stock is not comparable to a $50 move in another.
Normalization makes “which stock outperformed?” obvious.
Typical conclusion:

You can identify which stock had the strongest cumulative appreciation.
RSI reveals periods of stretched momentum.
Volume shows when market attention intensified.
Business/investor meaning:

Good first dashboard for performance leadership and momentum pressure.
Figure 2: Annual High / Low / Mean
See stock_analysis.py

What it shows:

For each stock, yearly maximum, minimum, and average close.
Use case:

Compare how each stock behaved year to year.
Understand whether gains came with wide trading ranges.
Why we need it:

It summarizes long time spans into a compact annual picture.
It helps detect whether a stock had stable appreciation or large swings.
Typical conclusion:

A rising mean with a widening high-low gap suggests strong growth with higher uncertainty.
A narrow spread suggests more stable price behavior.
Meaning:

Useful for annual trend review and risk framing.
Figure 3: Quarterly Analysis
See stock_analysis.py

What it shows:

Quarterly mean close price
Quarterly standard deviation as a volatility proxy
Use case:

Track performance at a finer level than annual summaries.
Spot periods where volatility surged in a particular quarter.
Why we need it:

Markets often change regime faster than yearly reporting can show.
Quarterly analysis helps tie movement to earnings cycles or macro events.
Typical conclusion:

Some quarters show strong price acceleration paired with rising volatility.
Others show smoother appreciation and calmer conditions.
Meaning:

Helps identify whether growth was orderly or unstable.
Figure 4: Up/Down Frequency and Daily Return Distributions
See stock_analysis.py

This is really two views.

First part:

Pie charts of up days vs down days
Use case:

Measure directional bias.
A stock with more up days than down days may show persistent bullish drift.
Second part:

Histogram + KDE of daily returns
Mean return marker
Skewness and kurtosis annotations
Use case:

Understand the shape of return behavior.
See whether the stock tends to have fat tails, asymmetry, or frequent large shocks.
Why we need it:

Average return alone is incomplete.
Two stocks can have similar averages but very different risk profiles.
Typical conclusion:

Positive skew suggests occasional strong upside jumps.
High kurtosis suggests extreme moves happen more often than normal.
A stock with many up days can still be risky if the down days are very large.
Meaning:

This figure is essential for risk understanding.
Figure 5: Correlation Heatmap
See stock_analysis.py

What it shows:

Correlation of returns
Correlation of prices
Use case:

Check how similarly these stocks move.
Important for diversification decisions.
Why we need it:

If all stocks move together, a portfolio of them gives less diversification than expected.
Return correlation is especially useful for portfolio construction.
Typical conclusion:

Tech stocks often show moderate to high co-movement.
Price correlation can look high simply because multiple stocks trend upward over time.
Return correlation is more informative for diversification.
Meaning:

Helps answer: “Are these actually giving me different exposure, or just more of the same?”
Figure 6: Bollinger Bands
See stock_analysis.py

What it shows:

Price relative to rolling upper and lower volatility bands
Use case:

Identify volatility expansion and possible overextension.
Price touching upper/lower bands may indicate strong momentum or stretched positioning.
Why we need it:

It combines price trend with volatility context.
A move is more meaningful when you know whether it is outside normal recent range.
Typical conclusion:

Repeated touches of the upper band indicate strong upward trend.
Sudden band widening indicates higher volatility regime.
Price crossing outside bands may suggest unusual movement.
Meaning:

Good for timing context and volatility-aware trend reading.
Figure 7: MACD
See stock_analysis.py

What it shows:

MACD and signal line, likely with momentum structure
Use case:

Detect trend acceleration or weakening.
Crossovers are often used as momentum signals.
Why we need it:

Price alone can lag momentum shifts.
MACD highlights changes in directional strength earlier than some slower indicators.
Typical conclusion:

MACD above signal suggests bullish momentum.
MACD below signal suggests bearish or weakening momentum.
Large separation indicates strong momentum.
Meaning:

Helps answer whether trend is strengthening or fading.
Figure 8: Rolling Volatility and Drawdown
See stock_analysis.py

What it shows:

30-day rolling volatility
Drawdown from historical peak
Use case:

Quantify risk dynamically.
Measure how painful the worst declines were.
Why we need it:

Investors care not only about return, but also how much the stock falls along the way.
Drawdown is one of the most intuitive risk measures.
Typical conclusion:

A stock may have great long-term return but still suffer deep temporary losses.
Volatility spikes often accompany market stress.
Bigger drawdowns imply higher emotional and capital risk.
Meaning:

Crucial for risk tolerance and portfolio suitability.
4. Machine learning setup
See stock_analysis.py

The notebook then switches from descriptive analytics to predictive modeling.

The build_features() function creates:

lagged closes: Lag1, Lag2, Lag3, Lag5, Lag10
moving averages: SMA5, SMA10
short-term return volatility: Vol5
technical features: MACD_f, RSI_f, BBW
Volume
Price_Range
The target variable is Close.

Why these features:

Lagged prices help models learn persistence and recent trend.
Moving averages help capture smoothing and local direction.
Technical indicators provide momentum and volatility context.
Volume and range add market activity information.
Train/test split:

It uses a chronological 80/20 split.
That is correct for time series compared with random shuffling.
Scaling:

Features are normalized with MinMaxScaler.
This helps models operate on comparable feature ranges.
Models used:

Linear Regression
Random Forest
Gradient Boosting
Why we need this section:

It tests whether recent market structure contains enough signal to estimate future close prices.
It compares a simple linear model against nonlinear ensemble models.
Conclusion:

This is not deep forecasting, but a reasonable baseline predictive experiment.
Figure 9: ML Predictions vs Actual
See stock_analysis.py

What it shows:

Actual close prices vs predicted close prices from each model for the test period
Use case:

Visual check of how closely each model tracks real movement.
Why we need it:

Metrics alone can hide pattern failures.
A chart reveals lagging behavior, smoothing, underreaction, or missed turning points.
Typical conclusion:

Linear Regression may track trend but miss nonlinear moves.
Random Forest and Gradient Boosting often fit market structure better.
If predictions are too smooth, the model is missing sudden shifts.
Meaning:

Good for seeing model behavior, not just numerical score.
Figure 10: Feature Importance
See stock_analysis.py

What it shows:

Random forest importance ranking for each feature
Use case:

Understand which inputs the model relied on most.
Why we need it:

ML without interpretability is harder to trust.
Importance shows whether price prediction is mostly driven by recent lags, volatility, volume, or indicators.
Typical conclusion:

Recent lag features usually dominate.
Technical features often contribute, but less than recent price memory.
If volume/range matter a lot, market activity is informative for that stock.
Meaning:

Helps explain the predictive structure rather than treating the model as a black box.
Figure 11: ML Metrics Heatmap
See stock_analysis.py

What it shows:

R² heatmap
RMSE heatmap
Use case:

Compare model quality across all stocks at once.
Why we need it:

One plot per stock is hard to summarize.
Heatmaps quickly show the best-performing model and hardest stock to predict.
How to interpret:

Higher R² is better.
Lower RMSE is better.
Typical conclusion:

Ensemble methods often outperform linear regression.
Some stocks are naturally easier to model because they have smoother trend structure.
More volatile or regime-switching stocks are harder to predict.
Meaning:

Best figure for model comparison at a glance.
Figure 12: Actual vs Predicted Scatter
See stock_analysis.py

What it shows:

Predicted values on one axis, actual values on the other
A reference diagonal line for perfect predictions
Use case:

Measure calibration visually.
If points lie close to the diagonal, predictions are good.
Why we need it:

Time-series line charts show tracking over time, but scatter plots show bias and dispersion.
You can spot systematic underprediction or overprediction.
Typical conclusion:

Tight clustering around the diagonal means strong model fit.
Wide scatter indicates unstable predictions.
If points consistently sit above or below the line, the model is biased.
Meaning:

Useful for validating whether the chosen “best” model is actually reliable.
Figure 13: Risk-Return Quadrant
See stock_analysis.py

What it shows:

Annualized return on one axis
Annualized risk on the other
Bubble size based on Sharpe-like ratio
Use case:

Compare reward versus risk across the stocks.
Why we need it:

Performance alone is incomplete.
The best investment is not always the highest return; it may be the best return per unit of risk.
Typical conclusion:

Stocks in the high-return, moderate-risk region are attractive.
Large bubble size suggests better risk-adjusted performance.
High return with very high risk may not be efficient.
Meaning:

This is a portfolio decision figure.
Figure 14: Monthly Return Heatmap
See stock_analysis.py

What it shows:

Average daily return by year and month for each stock
Use case:

Identify seasonality or recurring strong/weak months.
Why we need it:

Time aggregation can reveal recurring patterns hidden in daily noise.
Typical conclusion:

Some months may repeatedly show stronger or weaker performance.
Patterns may or may not be stable enough to act on.
Meaning:

Useful for exploratory seasonality analysis, but should be treated cautiously.
Figure 15: Statistical Summary Dashboard
See stock_analysis.py

What it shows:

Start price
End price
Total return
Mean return
Standard deviation
Sharpe ratio
Max drawdown
Skewness
Kurtosis
Up-days percentage
Use case:

One compact table to compare the whole stock set.
Why we need it:

After many charts, decision-makers often want one summary view.
This is the fastest way to compare overall performance and risk.
Typical conclusion:

You can see the tradeoff between total gain, consistency, asymmetry, and downside pain.
A stock with strong total return but deep drawdown may be less attractive than one with slightly lower return and much better Sharpe or drawdown profile.
