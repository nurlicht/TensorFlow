import tfjs from 'npm:@tensorflow/tfjs';
import { assert } from "https://deno.land/std@0.211.0/assert/mod.ts";
const codeAsString = Deno.readTextFileSync('./src/javascript/index.js');

Deno.test('Health-Check', async () => {
    assert((await new Promise((res, rej) => res(4))) === 4);
}, 1000);

Deno.test('TFJS app can be started and the fitted results are valid.', async () => {
    const tfjsApi = {tf: tfjs, tfvis: {render: {scatterplot: ()=>{}}}};
    const code = codeAsString + 'App.start(undefined, tfjsApi);';
    const result = await eval(code);
    assert(typeof result === 'object')
    assert(Array.isArray(result));
    assert(result.length === 2);
    assert(Array.isArray(result[0]));
    assert(Array.isArray(result[1]));
    assert(result[1].length === 400);
    assert(result[1][0].x === 0.5);
}, 50000);
