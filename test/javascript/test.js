const tfjs = require('@tensorflow/tfjs');
const AppModule = require('../../src/javascript/index.js');
const App = AppModule.App;
const InputDataParameters = AppModule.InputDataParameters;

test.concurrent('Health-Check', async () => {
    expect(await new Promise((res, rej) => res(4))).toBe(4);
}, 1000);

test.concurrent('TFJS app can be started and the fitted results are valid.', async () => {
    const inputDataParameters = InputDataParameters.default();
    const tfjsApi = {tf: tfjs, tfvis: {render: {scatterplot: ()=>{}}}};
    const result = await App.start(inputDataParameters, tfjsApi);
    expect(result).not.toBeNull();
    expect(Array.isArray(result)).toBeTruthy();
    expect(result.length).toBe(2);
    expect(Array.isArray(result[0])).toBeTruthy();
    expect(Array.isArray(result[1])).toBeTruthy();
    expect(result[1].length).toBe(inputDataParameters.nPoints);
    expect(result[1][0].x).toBe(inputDataParameters.minX);
}, 50000);
