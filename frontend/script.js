async function generate() {
    const prompt = document.getElementById('prompt').value;
    const duration = parseInt(document.getElementById('duration').value, 10);
    const statusEl = document.getElementById('status');
    const player = document.getElementById('player');
  
    statusEl.textContent = 'Generating...';
    player.src = '';
  
    try {
      const response = await fetch('http://127.0.0.1:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, duration })
      });
  
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Unknown error');
      }
  
      const data = await response.json();
      statusEl.textContent = data.message;
      player.src = `http://127.0.0.1:8000${data.audio_url}`;
    } catch (e) {
      statusEl.textContent = `Error: ${e.message}`;
    }
  }
  
  document.getElementById('generateBtn').addEventListener('click', generate);
  