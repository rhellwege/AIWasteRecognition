document.addEventListener('DOMContentLoaded', () => {
  const predictForm = document.getElementById('predictForm');
  const imageInput = document.getElementById('imageInput');
  const trainDiv = document.getElementById('trainDiv');

  trainDiv.style.display = "none";

  imageInput.addEventListener('change', () => {
    trainDiv.style.display = "block";
  });
  predictForm.addEventListener('submit', (event) => {
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
          console.log(data)
        })
    }

  });

  const organicBtn = document.getElementById('organic');
  const recyclableBtn = document.getElementById('recyclable');
  // training event listeners.
  organicBtn.addEventListener('click', () => {
    train("Organic")
  });
  recyclableBtn.addEventListener('click', () => {
    train("Recyclable")
  });


});

function train(label) {
  const payload = {
    method: "PUT",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ label: label })
  }
  fetch('/train', payload)
    .then(resp => {
      return resp.json();
    })
    .then(data => {
      console.log(data);
    })
    .catch(err => {
      console.error(err);
    });
}
