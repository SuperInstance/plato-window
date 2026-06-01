# plato-window

> Sliding window operations for PLATO tile streams with configurable padding strategies

## What This Does

plato-window provides sliding window operations over time series data. It supports configurable window size, stride, and four padding strategies (none, zero, mirror, wrap) for handling data boundaries. Includes windowed statistics: mean, sum, min, max, variance, and standard deviation.

## The Key Idea

Many signal processing operations need context: "What's the average of the last 10 readings?" A sliding window moves across the data, producing one output per position. The stride controls how far the window advances each step. Padding handles what happens at the edges where the window extends beyond the data.

## Install

```bash
cargo add plato-window
```

## Quick Start

```rust
use plato_window::*;

let config = WindowConfig::new(5)
    .with_stride(1)
    .with_padding(PaddingStrategy::Zero);

let windows = sliding_windows(&data, &config);
for w in &windows {
    println!("Mean: {:.2}, Std: {:.2}", window_mean(w), window_std(w));
}
```

## API Reference

| Type | Description |
|---|---|
| `WindowConfig { size, stride, padding }` | Builder: `new(size).with_stride(s).with_padding(p)` |
| `PaddingStrategy` | `None` / `Zero` / `Mirror` / `Wrap` |

### Functions

| Function | Description |
|---|---|
| `sliding_windows(data, config)` | Produce windows from data |
| `window_mean(window)` | Mean of window values |
| `window_sum(window)` | Sum of window values |
| `window_min(window)` / `window_max(window)` | Extremes |
| `window_variance(window)` / `window_std(window)` | Statistical measures |

## Testing

23 tests: window creation, stride, padding strategies, boundary handling, windowed statistics.

## License

Apache-2.0
