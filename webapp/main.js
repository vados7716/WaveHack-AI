async function fetchPrediction() {
  try {
    const response = await fetch('http://127.0.0.1:5000/predict', { method: 'POST' });
    const data = await response.json();

    if (response.ok) {
      const result = parseFloat(data.result).toFixed(2);
      const confidence = parseFloat(data.confidence).toFixed(2);

      document.getElementById('waitsvg').style.display = 'none';
      document.getElementById('result-text').textContent = `X ${result}`;
    } else {
      document.getElementById('result-text').textContent = `Error: ${data.error}`;
    }
  } catch (error) {
    document.getElementById('result-text').textContent = `Unexpected error: ${error.message}`;
  }
}
