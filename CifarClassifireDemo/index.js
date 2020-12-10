let testImg;
let imageRecieved = false;
let imageCanvas;
let canvasBackgroundColor = 100;
let predictButton;
let model;
let speech;
let predictionDisplay;
let testDisplayImage;
let labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
let testInput = new Array(32);
for (let i = 0; i < testInput.length; i++) {
  testInput[i] = new Array(32);
}
async function loadCifarModel() {
  model = await tf.loadModel("CifarModel/model.json");
}
function setup() {
  speech = new p5.Speech();
  loadCifarModel().then(() => predictionDisplay.html("Model loaded"));
  // create canvas 
  var c = createCanvas(windowWidth / 2 + 300, windowHeight / 2 + 400);
  imageCanvas = createGraphics(windowWidth / 2 + 300, windowHeight / 2 + 300);
  imageCanvas.background(0);
  background(100);
  // Add an event for when a file is dropped onto the canvas
  c.dragOver(function() {
    canvasBackgroundColor = 30;
  });
  c.dragLeave(function() {
    canvasBackgroundColor = 100;
  });
  console.log("changed2");
  c.drop(gotFile);
  predictButton = select("#predict_button");
  predictionDisplay = createP('');
  predictionDisplay.position(windowWidth / 2 + 300, (windowHeight / 2 + 300) + 80);
  //predictionDisplay.addClass('resultText');

  predictButton.mousePressed(predict);
  predictButton.position(windowWidth / 2 - 675, (windowHeight / 2 + 200) + 220);
}

function predict() {
  if (!imageRecieved) {
    return;
  }
  testImg.resize(32, 32);
  testImg.loadPixels();
  //pixel array = [R, G, B, A, R, G, B, A.........]
  for (let y = 0; y < testImg.height; y++) {
    for (let x = 0; x < testImg.width; x++) {
      let index = (x + y * testImg.width) * 4;
      //console.log(index);
      let r = testImg.pixels[index] / 255;
      let g = testImg.pixels[index + 1] / 255;
      let b = testImg.pixels[index + 2] / 255;
      let a = testImg.pixels[index + 3] / 255;
      // console.log("R="+r);
      // console.log("G="+g);
      // console.log("B="+b);
      // console.log("A="+a);
      let rgbArray = [r, g, b];
      testInput[y][x] = rgbArray;
      // console.log(rgbArray);
    }
  }
  //console.log("("+testInput.length+", "+testInput[0].length+", "+testInput[0][0].length+")");
  let testInput4d = [testInput];
  //console.log("("+testInput4d.length+", "+testInput4d[0].length+", "+testInput4d[0][0].length+", "+testInput4d[0][0][0].length+")");
  let testInputTensor = tf.tensor4d(testInput4d);
  let prediction = model.predict(testInputTensor).dataSync();
  let maxIndex = argmax(prediction);
  let vowel = 'a';
  if (maxIndex == 0 || maxIndex == 1) {
    vowel = 'an';
  }
  console.log("Label:"+labels[maxIndex]);
  speech.speak("Is it "+vowel+" "+labels[maxIndex]);
  predictionDisplay.html("Is it "+vowel+" "+labels[maxIndex]);
  // console.log(testInputTensor.shape);
  // console.log(testInputTensor.print());
  testImg.resize(width, height);
}

function argmax(arr) {
  let maxIndex = 0;
  let max = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      max = arr[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

function draw() {
  background(canvasBackgroundColor);
  image(imageCanvas, 0, 0);
  if (imageRecieved) {
    imageCanvas.image(testDisplayImage, 0, 0, imageCanvas.width, imageCanvas.height);
  }
  fill(255);
  noStroke();
  textSize(40);
  textAlign(CENTER);
  text('Drag an image file here.', width / 2, height - 40);
  console.log("1231");
}

function gotFile(file) {
  // If it's an image file
  if (file.type === 'image') {
    // Create an image DOM element but don't show it
    testImg = loadImage(file.data);
    testDisplayImage = loadImage(file.data);
    canvasBackgroundColor = 100;
    imageRecieved = true;
    // Draw the image onto the canvas
    //image(img, 0, 0, width, height);
  } else {
    println('Not an image file!');
    imageRecieved = false;
  }

}


function mouseDragged() {
  console.log('qewe');
  let diffX = pmouseX - mouseX;
  let diffY = pmouseY - mouseY;
  if (diffX < 0) {
    console.log("right");
  }
}
