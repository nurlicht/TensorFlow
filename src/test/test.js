const tfjs = require('@tensorflow/tfjs');
const App = require('../main/javascript/index.js');

test.concurrent('Health-Check', async () => {
    expect(await new Promise((res, rej) => res(4))).toBe(4);
}, 1000);

test.concurrent('TFJS app can be started and the fitted results are valid.', async () => {
    const result = await App.start({tf: tfjs, tfvis: {render: {scatterplot: ()=>{}}}});
    expect(result).not.toBeNull();
    expect(Array.isArray(result)).toBeTruthy();
    expect(result.length).toBe(2);
    expect(Array.isArray(result[0])).toBeTruthy();
    expect(Array.isArray(result[1])).toBeTruthy();
    expect(result[1].length).toBe(400);
    expect(result[1][0].x).toBe(0.5);
}, 50000);
