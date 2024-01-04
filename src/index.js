class TfjsApi {
  // https://js.tensorflow.org/api/latest/
  static tf = tf;

  // https://js.tensorflow.org/api_vis/latest/
  static tfvis = tfvis; 
}

class Utilities {
  static removeUndefinedParameters(x) {
    const reducedX = {};
    Object.keys(x)
      .filter((key) => typeof key === 'string')
      .forEach((key) => reducedX[key] = x[key]);
    return reducedX;
  }

  static positiveInteger(x, xDefault) {
    return (Number.isInteger(x) && x > 0) ? x : xDefault;
  }

  static string(x, xDefault) {
    return (typeof x === 'string') ? x : xDefault;
  }

  /**
    Goal: Generation of a 'sparse' version of a vector 0:xMax (with a control parameter 'k')
    Returns true if 'x' (from [0 xMax]) should be preserved.
    (Nearly) Decreasing function of 'k' (for small 'x')
    Different behavior for small and large 'x' because of the chirp-factor p*p (for a given 'k')
  */
  static sparse(x, xMax, k) {
    const p = (x / xMax);
    return parseInt(x + 31.0 * k * k * p * p) % k === 0;
  }

  static validateArrayPair(x, y) {
    if (!Array.isArray(x) || !Array.isArray(y)) {
      throw new Error('Non-array objects were encountered.');
    }
    if (x.length !== y.length) {
      throw new Error('Arrays with different lengths were encountered.');
    }
    if ([...x, ...y].filter((p) => typeof p !== 'number').length > 0) {
      throw new Error('Arrays with non-number elements were encountered.');
    }
  }

  static toPointCloud(x, y) {
    Utilities.validateArrayPair(x, y);
    const indices = Array(x.length).fill().map((element, index) => index);
    TfjsApi.tf.tidy(() => TfjsApi.tf.util.shuffle(indices));
    return indices.map((index) => {return {'x': x[index], 'y': y[index]};});
  }
}

class ArrayWithMinMax {
  array;
  tensor;
  min;
  max;

  constructor(array) {
    this.array = array;
    this.tensor = TfjsApi.tf.tensor2d(this.array, [this.array.length, 1]);
    this.min = this.tensor.min();
    this.max = this.tensor.max();
  }

  static create(x, indices) {
    return new ArrayWithMinMax(indices.map((index) => x[index]));
  }

  normalize() {
    return TfjsApi.tf.tidy(() => this.tensor.sub(this.min).div(this.max.sub(this.min)));
  }

  scale(x) {
    return TfjsApi.tf.tidy(() => x.mul(this.max.sub(this.min)).add(this.min));
  }

  static createPair(x, y) {
    Utilities.validateArrayPair(x, y);
    const indices = Array(x.length).fill().map((element, index) => index);
    return TfjsApi.tf.tidy(() => {
      TfjsApi.tf.util.shuffle(indices);
      const X = ArrayWithMinMax.create(x, indices);
      const Y = ArrayWithMinMax.create(y, indices);
      return { X, Y };
    });
  }
}

class InputDataParameters {
  minX = 0.5;
  maxX = 4.0;
  nPoints = 400;
  coefficient = 2.0;
  noisePercent = 2.5;
  sparsity = 2; //1-6
  batchSize = 32;
  epochs = 200;
}

class ModelData {
  x;
  y;
  params;

  static create(inputDataParameters) {
    const instance = new ModelData();
    console.log('inputDataParameters', inputDataParameters);
    instance.set(inputDataParameters);
    return instance;
  }

  set(inputDataParameters) {
    this.setX(inputDataParameters);
    this.setY(inputDataParameters);
    this.params = inputDataParameters;
  }

  setX(inputDataParameters) {
    const xFactor = (inputDataParameters.maxX - inputDataParameters.minX) /
      (inputDataParameters.nPoints- 1);
    this.x = Array(inputDataParameters.nPoints)
      .fill()
      .map((element, index) => inputDataParameters.minX + xFactor * index)
      .filter((element, index) =>
        Utilities.sparse(index, inputDataParameters.nPoints - 1, inputDataParameters.sparsity))
      ;
  }

  setY(inputDataParameters) {
    const y = this.x.map((p) => Math.exp(- inputDataParameters.coefficient * p));
    const maxY = Math.max(...y);
    const minY = Math.min(...y);
    const noiseFactor = (inputDataParameters.noisePercent / 100.0) * (maxY - minY) / (0.5 * (maxY + minY));
    this.y = y.map((p) => p + noiseFactor * Math.random());
  }
}

class ModelProvider {
  modelLayerParametersList;
  model;
 
  constructor() {
    this.set();
  }

  set(modelLayerParametersList) {
    this.modelLayerParametersList = Array.isArray(modelLayerParametersList) ?
      modelLayerParametersList :
      ModelProvider.getDefaultModelLayerParametersList();
    this.model = TfjsApi.tf.sequential();
    this.modelLayerParametersList.forEach((param) => this.model.add(TfjsApi.tf.layers.dense(param)));
  }

  static getDefaultModelLayerParametersList() {
    return [
      {inputShape: [1], units: 50, useBias: true}, // Input
      {units: 50, activation: 'sigmoid'},          // Middle
      {units: 1, useBias: true}                    // Output
    ];
  }
}

class Trainer {
  static async train(model, data, inputDataParameters) {
    model.compile({
      optimizer: TfjsApi.tf.train.adam(),
      loss: TfjsApi.tf.losses.meanSquaredError,
      metrics: ['mse'],
    });

    const onEpochEnd = (epoch, logs) => console.log('mse', logs.mse);

    await model.fit(data.X.normalize(), data.Y.normalize(), {
      batchSize: inputDataParameters.batchSize,
      epochs: inputDataParameters.epochs,
      shuffle: true,
      callbacks: {onEpochEnd}
    });
  }

  static getTfvisCallback() {
    return TfjsApi.tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 250, callbacks: ['onEpochEnd'] }
    );
  }

  static async test(model, data, inputDataParameters) {
    const [xScaled, yScaledEstimated] = TfjsApi.tf.tidy(() => {
      const nPoints = inputDataParameters.nPoints;
      const xNorm = TfjsApi.tf.linspace(0, 1, nPoints);
      const yNormEstimated = model.predict(xNorm.reshape([nPoints, 1]));
      return [data.X.scale(xNorm).dataSync(), data.Y.scale(yNormEstimated).dataSync()];
    });

    const pointCloudEstimated = Array.from(xScaled).map((element, index) => {
      return {x: element, y: yScaledEstimated[index]}
    });
    const pointCloudOriginal = data.X.array.map((element, index) => ({
      x: element, y: data.Y.array[index],
    }));
    View.showScatterPlot([pointCloudOriginal, pointCloudEstimated], inputDataParameters);
  }
}

class View {
  static showModelSummary(model) {
    TfjsApi.tfvis.show.modelSummary(
      {name: 'Summary of Model'},
      model
    );
  }

  static showScatterPlot(values, inputDataParameters, xLabel, yLabel, height) {
    const hasMultipleCurves = values.length > 0 && Array.isArray(values[0]);
    const data = hasMultipleCurves ?
      {values, series: ['Actual', 'Predicted']} :
      {values, series: ['Actual']};
    const nameExtension = typeof inputDataParameters === 'object' ?
      `(noise=${inputDataParameters.noisePercent}%, epochs=${inputDataParameters.epochs})` :
      '';
    TfjsApi.tfvis.render.scatterplot(
      { name: `Output vs. Input ${nameExtension}`},
      data,
      {
        xLabel: Utilities.string(xLabel, 'Input'),
        yLabel: Utilities.string(yLabel, 'Output'),
        height: Utilities.positiveInteger(height, 140)
      }
    );
  }
}

class Controller {
  static async run(inputDataParameters) {
    // Import and visualize input-data
    const modelData = ModelData.create(inputDataParameters);
    modelData.set(inputDataParameters);
    console.log('modelData', modelData)
    const modelDataAsPointCloud = Utilities.toPointCloud(modelData.x, modelData.y);
    View.showScatterPlot(modelDataAsPointCloud);

    // Define and display the TensorFlow-model
    const modelProvider = new ModelProvider();
    //View.showModelSummary(modelProvider.model);

    // Train and test the model
    const tensors = ArrayWithMinMax.createPair(modelData.x, modelData.y);
    await Trainer.train(modelProvider.model, tensors, inputDataParameters);
    await Trainer.test(modelProvider.model, tensors, inputDataParameters);
  }

  static createSliders(inputDataParameters) {
    let sliderIndex = 0;
    const sliders = {
      noisePercent: Controller.createSlider('Noise', 0, inputDataParameters.noisePercent * 2, sliderIndex++),
      epochs: Controller.createSlider('Epoch', 100, inputDataParameters.epochs * 2, sliderIndex++)
    };
    Controller.setSliderCallbacks(sliders, inputDataParameters);
    return sliders;
  }

  static createSlider(label, min, max, index) {
    const slider = document.createElement('input');
    slider.id = `slider-${index}`;
    slider.type = 'range';
    slider.class = 'slider';
    slider.min = min;
    slider.max = max;
    slider.value = (min + max) / 2;
  
    const sliderLabel = document.createElement('label');
    sliderLabel.for = slider.id;
    sliderLabel.innerHTML = label;

    const visor = document.getElementsByClassName('visor-surfaces')[0];
    const sliderContainer = document.createElement('div');
    sliderContainer.style.marginBottom = '20px';
    visor.appendChild(sliderContainer);
    sliderContainer.appendChild(sliderLabel);
    sliderContainer.appendChild(slider);

    return slider; 
  }

  static disableSilders(sliders) {
    Object.keys(sliders).forEach((key) => sliders[key].disabled = true);
  }

  static enableSilders(sliders) {
    Object.keys(sliders).forEach((key) => sliders[key].disabled = false);
  }

  static setSliderCallbacks(sliders, inputDataParameters) {
    Object.keys(sliders).forEach((key) => {
      sliders[key].oninput = async () => {
        Controller.disableSilders(sliders);
        inputDataParameters[key] = sliders[key].value;
        await Controller.run(inputDataParameters);
        Controller.enableSilders(sliders);
      }
    });
  }

  static async start(sliders) {
    await sliders.epochs.oninput();
  }
}

class App {
  static async run() {
    // Initialize the parameters
    const inputDataParameters = new InputDataParameters();

    // Initialize the scatter-plot
    View.showScatterPlot([]);

    //Define sliders
    const sliders = Controller.createSliders(inputDataParameters);

    // Trigger a new fit
    Controller.start(sliders);
  }

  static runAfterLoadingTfjs() {
    document.addEventListener('DOMContentLoaded', App.run);
  }

  static start() {
    App.runAfterLoadingTfjs();
  }
}
