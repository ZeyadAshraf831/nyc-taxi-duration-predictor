const state = {
  pickingMode: "pickup",
  pickup: null,
  dropoff: null,
};

const pickupIcon = L.divIcon({
  className: "map-marker pickup-marker",
  html: '<div class="marker-dot pickup"></div>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

const dropoffIcon = L.divIcon({
  className: "map-marker dropoff-marker",
  html: '<div class="marker-dot dropoff"></div>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

const map = L.map("map").setView(
  [window.APP_CONFIG.mapLat, window.APP_CONFIG.mapLng],
  window.APP_CONFIG.mapZoom
);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);

let pickupMarker = null;
let dropoffMarker = null;
let routeLine = null;

const pickupModeBtn = document.getElementById("pickup-mode-btn");
const dropoffModeBtn = document.getElementById("dropoff-mode-btn");
const mapHint = document.getElementById("map-hint");
const pickupStatus = document.getElementById("pickup-status");
const dropoffStatus = document.getElementById("dropoff-status");
const tripForm = document.getElementById("trip-form");
const passengerCount = document.getElementById("passenger-count");
const passengerValue = document.getElementById("passenger-value");
const pickupDate = document.getElementById("pickup-date");
const pickupHour = document.getElementById("pickup-hour");
const pickupMinute = document.getElementById("pickup-minute");
const predictBtn = document.getElementById("predict-btn");
const resultBox = document.getElementById("result");
const resultMessage = document.getElementById("result-message");
const errorBox = document.getElementById("error");
const metricDistance = document.getElementById("metric-distance");
const metricRush = document.getElementById("metric-rush");
const metricWeekend = document.getElementById("metric-weekend");

function setMode(mode) {
  state.pickingMode = mode;
  pickupModeBtn.classList.toggle("active", mode === "pickup");
  dropoffModeBtn.classList.toggle("active", mode === "dropoff");
  mapHint.textContent =
    mode === "pickup"
      ? "Click on the map to set your pickup location."
      : "Click on the map to set your dropoff location.";
}

function updateStatusCard(element, label, point) {
  if (!point) {
    element.className = "status-card warning";
    element.textContent = `${label}: Not selected`;
    return;
  }

  element.className = "status-card success";
  element.textContent = `${label}: ${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}`;
}

function updateRouteLine() {
  if (routeLine) {
    map.removeLayer(routeLine);
    routeLine = null;
  }

  if (!state.pickup || !state.dropoff) {
    return;
  }

  routeLine = L.polyline(
    [
      [state.pickup.lat, state.pickup.lng],
      [state.dropoff.lat, state.dropoff.lng],
    ],
    { color: "#2563eb", weight: 4, opacity: 0.75 }
  ).addTo(map);
}

function setPickup(latlng) {
  state.pickup = { lat: latlng.lat, lng: latlng.lng };

  if (pickupMarker) {
    pickupMarker.setLatLng(latlng);
  } else {
    pickupMarker = L.marker(latlng, { icon: pickupIcon }).addTo(map);
  }

  updateStatusCard(pickupStatus, "Pickup", state.pickup);
  updateRouteLine();
  setMode("dropoff");
}

function setDropoff(latlng) {
  state.dropoff = { lat: latlng.lat, lng: latlng.lng };

  if (dropoffMarker) {
    dropoffMarker.setLatLng(latlng);
  } else {
    dropoffMarker = L.marker(latlng, { icon: dropoffIcon }).addTo(map);
  }

  updateStatusCard(dropoffStatus, "Dropoff", state.dropoff);
  updateRouteLine();
}

function initDateTimeFields() {
  const now = new Date();
  pickupDate.value = now.toISOString().slice(0, 10);

  for (let hour = 0; hour < 24; hour += 1) {
    const option = document.createElement("option");
    option.value = String(hour);
    option.textContent = String(hour).padStart(2, "0");
    if (hour === now.getHours()) {
      option.selected = true;
    }
    pickupHour.appendChild(option);
  }

  for (let minute = 0; minute < 60; minute += 1) {
    const option = document.createElement("option");
    option.value = String(minute);
    option.textContent = String(minute).padStart(2, "0");
    if (minute === now.getMinutes()) {
      option.selected = true;
    }
    pickupMinute.appendChild(option);
  }
}

function hideMessages() {
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
}

function showResult(data) {
  resultMessage.textContent = `Estimated trip duration: ${data.minutes} min ${data.seconds} sec`;
  metricDistance.textContent = `${data.distance_km} km`;
  metricRush.textContent = data.is_rush_hour ? "Yes" : "No";
  metricWeekend.textContent = data.is_weekend ? "Yes" : "No";
  resultBox.classList.remove("hidden");
  errorBox.classList.add("hidden");
}

pickupModeBtn.addEventListener("click", () => setMode("pickup"));
dropoffModeBtn.addEventListener("click", () => setMode("dropoff"));

passengerCount.addEventListener("input", () => {
  passengerValue.textContent = passengerCount.value;
});

map.on("click", (event) => {
  if (state.pickingMode === "pickup") {
    setPickup(event.latlng);
  } else {
    setDropoff(event.latlng);
  }
});

tripForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideMessages();

  if (!state.pickup || !state.dropoff) {
    showError("Please select both pickup and dropoff locations on the map before predicting.");
    return;
  }

  const hour = String(pickupHour.value).padStart(2, "0");
  const minute = String(pickupMinute.value).padStart(2, "0");
  const pickupDatetime = `${pickupDate.value}T${hour}:${minute}:00`;

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pickup_lat: state.pickup.lat,
        pickup_lon: state.pickup.lng,
        dropoff_lat: state.dropoff.lat,
        dropoff_lon: state.dropoff.lng,
        vendor_id: Number(document.getElementById("vendor-id").value),
        passenger_count: Number(passengerCount.value),
        pickup_datetime: pickupDatetime,
      }),
    });

    const data = await response.json();
    if (!response.ok || !data.success) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    showResult(data);
  } catch (error) {
    showError("Unable to reach the prediction service. Please try again.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Trip Duration";
  }
});

initDateTimeFields();
setMode("pickup");
