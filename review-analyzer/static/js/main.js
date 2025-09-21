// static/js/main.js
document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const review = document.getElementById("reviewInput").value.trim();
  if (!review) {
    alert("Please enter a review.");
    return;
  }
  const resDiv = document.getElementById("result");
  const predSpan = document.getElementById("prediction");
  const probsDiv = document.getElementById("probs");
  resDiv.classList.add("hidden");

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review })
    });
    const data = await resp.json();
    if (resp.ok) {
      predSpan.textContent = data.prediction;
      probsDiv.innerHTML = "";
      if (data.probabilities && Object.keys(data.probabilities).length > 0) {
        const ul = document.createElement("ul");
        for (const [cls, p] of Object.entries(data.probabilities)) {
          const li = document.createElement("li");
          li.textContent = `${cls}: ${(p*100).toFixed(2)}%`;
          ul.appendChild(li);
        }
        probsDiv.appendChild(ul);
      }
      resDiv.classList.remove("hidden");
    } else {
      alert(data.error || "Error analyzing review");
    }
  } catch (err) {
    console.error(err);
    alert("Server error. See console for details.");
  }
});
