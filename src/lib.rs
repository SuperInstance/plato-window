use serde::{Deserialize, Serialize};

/// Strategy for padding data when the window extends beyond boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// No padding; windows must fit within data bounds.
    None,
    /// Pad with zeros.
    Zero,
    /// Mirror the data at boundaries.
    Mirror,
    /// Wrap around (circular).
    Wrap,
}

impl PaddingStrategy {
    /// Apply padding to `data` so that a window of `window_size` can slide
    /// from the first element. Returns a padded copy of `data`.
    pub fn apply(&self, data: &[f64], window_size: usize) -> Vec<f64> {
        if data.is_empty() || window_size <= data.len() {
            return data.to_vec();
        }
        let pad = window_size - data.len();
        match self {
            PaddingStrategy::None => data.to_vec(),
            PaddingStrategy::Zero => {
                let mut out = vec![0.0; pad];
                out.extend_from_slice(data);
                out
            }
            PaddingStrategy::Mirror => {
                let mut out = Vec::with_capacity(window_size);
                // prepend reversed first `pad` elements
                for i in (0..pad).rev() {
                    out.push(data[i]);
                }
                out.extend_from_slice(data);
                out
            }
            PaddingStrategy::Wrap => {
                let mut out = Vec::with_capacity(window_size);
                // wrap from the start of data
                for i in 0..pad {
                    out.push(data[i % data.len()]);
                }
                out.extend_from_slice(data);
                out
            }
        }
    }
}

/// Configuration for sliding window operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    pub size: usize,
    pub stride: usize,
    pub padding: PaddingStrategy,
}

impl WindowConfig {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            stride: 1,
            padding: PaddingStrategy::None,
        }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride.max(1);
        self
    }

    pub fn with_padding(mut self, padding: PaddingStrategy) -> Self {
        self.padding = padding;
        self
    }
}

/// Statistical summary of a window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowStats {
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub skewness: f64,
}

/// Sliding window over a 1D data stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Window<T> {
    data: Vec<T>,
    size: usize,
    position: usize,
}

impl Window<f64> {
    /// Create a new sliding window with the given data and window size.
    pub fn new(data: Vec<f64>, size: usize) -> Self {
        Self {
            data,
            size: size.max(1),
            position: 0,
        }
    }

    /// Create a window with full configuration.
    pub fn with_config(data: Vec<f64>, config: WindowConfig) -> Self {
        let padded = config.padding.apply(&data, config.size);
        Self {
            data: padded,
            size: config.size.max(1),
            position: 0,
        }
    }

    /// Advance the window position by `stride`. Returns `false` if at the end.
    pub fn slide(&mut self, stride: usize) -> bool {
        let stride = stride.max(1);
        self.position += stride;
        self.position + self.size <= self.data.len()
    }

    /// Get the current window contents.
    pub fn current(&self) -> &[f64] {
        let end = (self.position + self.size).min(self.data.len());
        &self.data[self.position..end]
    }

    /// Collect all windows according to config.
    pub fn collect_all(data: Vec<f64>, config: WindowConfig) -> Vec<Vec<f64>> {
        let padded = config.padding.apply(&data, config.size);
        let size = config.size.max(1);
        let stride = config.stride.max(1);
        let mut windows = Vec::new();
        if padded.len() < size {
            return windows;
        }
        let mut pos = 0;
        while pos + size <= padded.len() {
            windows.push(padded[pos..pos + size].to_vec());
            pos += stride;
        }
        windows
    }

    /// Compute statistics on the current window.
    pub fn stats(&self) -> WindowStats {
        let w = self.current();
        let n = w.len() as f64;
        if n == 0.0 {
            return WindowStats {
                mean: 0.0,
                variance: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                skewness: 0.0,
            };
        }

        let mut sorted = w.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = w.iter().sum();
        let mean = sum / n;

        let variance = if n > 1.0 {
            w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let std_dev = variance.sqrt();
        let skewness = if std_dev > 0.0 && n > 2.0 {
            let m3: f64 = w.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum();
            m3 * n / ((n - 1.0) * (n - 2.0))
        } else {
            0.0
        };

        WindowStats {
            mean,
            variance,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            median,
            skewness,
        }
    }

    /// Get the window at a specific position.
    pub fn at(&self, i: usize) -> Option<&[f64]> {
        let start = i;
        let end = start + self.size;
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Total number of non-overlapping positions (stride=1).
    pub fn count(&self) -> usize {
        if self.data.len() >= self.size {
            self.data.len() - self.size + 1
        } else {
            0
        }
    }
}

/// 1D convolution of data with a kernel.
pub fn convolve(data: &[f64], kernel: &[f64]) -> Vec<f64> {
    if kernel.is_empty() || data.len() < kernel.len() {
        return vec![];
    }
    let k_len = kernel.len();
    let out_len = data.len() - k_len + 1;
    let mut result = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let val: f64 = data[i..i + k_len]
            .iter()
            .zip(kernel.iter())
            .map(|(d, k)| d * k)
            .sum();
        result.push(val);
    }
    result
}

/// Simple moving average.
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Moving standard deviation (sample, ddof=1).
pub fn moving_std(data: &[f64], window: usize) -> Vec<f64> {
    if window < 2 || data.len() < window {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=data.len() - window {
        let w = &data[i..i + window];
        let n = window as f64;
        let mean: f64 = w.iter().sum::<f64>() / n;
        let var: f64 = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        result.push(var.sqrt());
    }
    result
}

/// Moving maximum.
pub fn moving_max(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=data.len() - window {
        let w = &data[i..i + window];
        result.push(w.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    }
    result
}

/// Moving minimum.
pub fn moving_min(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=data.len() - window {
        let w = &data[i..i + window];
        result.push(w.iter().cloned().fold(f64::INFINITY, f64::min));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sliding_window() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut w = Window::new(data, 3);
        assert_eq!(w.current(), &[1.0, 2.0, 3.0]);
        assert!(w.slide(1));
        assert_eq!(w.current(), &[2.0, 3.0, 4.0]);
        assert!(w.slide(1));
        assert_eq!(w.current(), &[3.0, 4.0, 5.0]);
        assert!(!w.slide(1));
    }

    #[test]
    fn test_stride_greater_than_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut w = Window::new(data, 3);
        assert_eq!(w.current(), &[1.0, 2.0, 3.0]);
        assert!(w.slide(2));
        assert_eq!(w.current(), &[3.0, 4.0, 5.0]);
        assert!(!w.slide(2));
    }

    #[test]
    fn test_padding_zero() {
        let data = vec![1.0, 2.0];
        let padded = PaddingStrategy::Zero.apply(&data, 4);
        assert_eq!(padded, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_padding_mirror() {
        let data = vec![1.0, 2.0, 3.0];
        let padded = PaddingStrategy::Mirror.apply(&data, 5);
        assert_eq!(padded, vec![2.0, 1.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_padding_wrap() {
        let data = vec![1.0, 2.0, 3.0];
        let padded = PaddingStrategy::Wrap.apply(&data, 5);
        assert_eq!(padded, vec![1.0, 2.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_padding_none_when_fits() {
        let data = vec![1.0, 2.0, 3.0];
        let padded = PaddingStrategy::None.apply(&data, 2);
        assert_eq!(padded, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_window_stats_mean() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let w = Window::new(data, 5);
        let stats = w.stats();
        assert!((stats.mean - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_stats_variance() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let w = Window::new(data, 5);
        let stats = w.stats();
        // sample variance: sum((xi - mean)^2) / (n-1) = 40/4 = 10
        assert!((stats.variance - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_stats_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.5, 9.0];
        let w = Window::new(data, 5);
        let stats = w.stats();
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_stats_median_odd() {
        let data = vec![1.0, 3.0, 5.0];
        let w = Window::new(data, 3);
        let stats = w.stats();
        assert!((stats.median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_stats_median_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let w = Window::new(data, 4);
        let stats = w.stats();
        assert!((stats.median - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_collect_all_windows() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let config = WindowConfig::new(2);
        let windows = Window::collect_all(data, config);
        assert_eq!(windows, vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
        ]);
    }

    #[test]
    fn test_collect_all_with_stride() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = WindowConfig::new(2).with_stride(2);
        let windows = Window::collect_all(data, config);
        assert_eq!(windows, vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
    }

    #[test]
    fn test_convolve_simple() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0];
        let result = convolve(&data, &kernel);
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = moving_average(&data, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_std() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = moving_std(&data, 3);
        // manually: [2,4,4] -> mean=3.33, var=1.33, std=1.15
        assert!(!result.is_empty());
        assert!(result[0] > 0.0);
    }

    #[test]
    fn test_moving_max() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let result = moving_max(&data, 3);
        assert_eq!(result, vec![5.0, 5.0, 4.0]);
    }

    #[test]
    fn test_moving_min() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let result = moving_min(&data, 3);
        assert_eq!(result, vec![1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_edge_window_equals_data() {
        let data = vec![1.0, 2.0, 3.0];
        let mut w = Window::new(data, 3);
        assert_eq!(w.current(), &[1.0, 2.0, 3.0]);
        assert!(!w.slide(1));
    }

    #[test]
    fn test_edge_window_larger_than_data() {
        let data = vec![1.0, 2.0];
        let w = Window::new(data, 5);
        assert!(w.current().len() < 5); // partial or empty
        assert_eq!(w.count(), 0);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<f64> = vec![];
        let result = moving_average(&data, 3);
        assert!(result.is_empty());
        let result = convolve(&data, &[1.0]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_skewness() {
        // symmetric data should have ~0 skewness
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = Window::new(data, 5);
        let stats = w.stats();
        assert!(stats.skewness.abs() < 0.01);
    }

    #[test]
    fn test_position_access() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let w = Window::new(data, 2);
        assert_eq!(w.at(0), Some(&[10.0, 20.0][..]));
        assert_eq!(w.at(3), Some(&[40.0, 50.0][..]));
        assert_eq!(w.at(4), None);
        assert_eq!(w.count(), 4);
    }
}
