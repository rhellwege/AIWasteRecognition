document.addEventListener('DOMContentLoaded', () => {
  const predictForm = document.getElementById('predictForm');
  const imageInput = document.getElementById('imageInput');
  const trainDiv = document.getElementById('trainDiv');
  const predictionH2 = document.getElementById('currentPrediction');

  trainDiv.style.display = "none";

  imageInput.addEventListener('change', () => {
    if (imageInput.files.length > 0) {
      trainDiv.style.display = "block";
    }
    else {
      trainDiv.style.display = "none";
    }
  });
  peedictForm.addEventListener('submit', (event) => {
    event.preventDefault(); // do not redirect to a seperate page, we want to hit a JSON endpoint.
    // manually populate the form and send the data to the server, expecting json response.
    if (imageInput.files.length > 0) {
      const imageFile = imageInput.files[0];
      const formData = new FormData();
      formData.append('image', imageFile);
      fetch('/predict', {
        method: 'POST',
        body: formData
      })
        .then(resp => {
          return resp.json();
        })
        .then(data => {
          const percentage = (Math.max(data.probabilities[0], data.probabilities[1]) * 100).toFixed(2);
          predictionH2.innerText = `I am ${percentage}% sure that this is ${data.prediction}`;
          console.log(data);
        })
    }
  });

  const reloadBtn = document.getElementById('reloadButton');
  reloadBtn.addEventListener('click', () => {
    fetch('/reload-model', {
      method: 'PUT'
    })
      .then(resp => resp.text())
      .then(data => console.log(data));
  });

  const organicBtn = document.getElementById('organic');
  const recyclableBtn = document.getElementById('recyclable');
  // training event listeners.
  organicBtn.addEventListener('click', () => {
    train(imageInput, "Organic")
  });
  recyclableBtn.addEventListener('click', () => {
    train(imageInput, "Recyclable")
  });

});

function train(imageInput, label) {
  if (imageInput.files.length > 0) {
    const imageFile = imageInput.files[0];
    const formData = new FormData();
    const trainH3 = document.getElementById('currentLoss');
    formData.append('image', imageFile);
    formData.append('label', label)
    fetch('/train', {
      method: 'PUT',
      body: formData
    })
      .then(resp => {
        return resp.json();
      })
      .then(data => {
        trainH3.innerText = `Training Loss: ${data.loss.toFixed(4)}`;
        console.log(data);
      })
  }
}
