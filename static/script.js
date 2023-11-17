document.addEventListener('DOMContentLoaded', () => {
  const organicBtn = document.getElementById('organic');
  const recyclableBtn = document.getElementById('recyclable');
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
