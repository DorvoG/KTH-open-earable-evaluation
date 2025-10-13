const appUI = document.querySelector(".ui");
const bpmTxt = document.querySelector(".bpm");
const connectBTN = document.querySelector(".connect");
const connectUI = document.querySelector(".connect-ui");  //takes elements from HTML

let device;
let heartRate;

function parseHeartRate(value) {                          //decodes raw bluetooth data
  let is16Bits = value.getUint8(0) & 0x1;
  if (is16Bits) return value.getUint16(1, true);
  return value.getUint8(1);
}

function handleRateChange(event) {                            //updates web page when data changes
  bpmTxt.textContent = parseHeartRate(event.target.value);
}

async function requestDevice() {                                    //finds the bluetooth device
  const options = {acceptAllDevices: true};
  device = await navigator.bluetooth.requestDevice(options);
  device.addEventListener("gattserverdisconnected", connectDevice);
}

async function connectDevice() {                                  //connects to the devices GATT server
  if (device.gatt.connected) return;

  const server = await device.gatt.connect();
  const service = await server.getPrimaryService("heart_rate");

  heartRate = await service.getCharacteristic("heart_rate_measurement");      //listens for heart rate value from sensor
  heartRate.addEventListener("characteristicvaluechanged", handleRateChange);
  console.log("connected");
}

async function init() {                                                     //main init flow
  if (!navigator.bluetooth) return errorTxt.classList.remove("hide");
  if (!device) await requestDevice();

  connectBTN.textContent = "connecting...";
  await connectDevice();

  appUI.classList.remove("hide");
  connectUI.classList.add("hide");
  await startMonitoring();
}

connectBTN.addEventListener("click", init);
