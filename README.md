# Interactive data generation and fitting [![](https://github.com/nurlicht/TensorFlow/actions/workflows/node.js.yml/badge.svg)](https://github.com/nurlicht/TensorFlow/actions)[![](https://github.com/nurlicht/TensorFlow/actions/workflows/deno.yml/badge.svg)](https://github.com/nurlicht/TensorFlow/actions)

### Background
See the codelab [TensorFlow.js — Making Predictions from 2D Data](https://codelabs.developers.google.com/codelabs/tfjs-training-regression) for an introduction to TenosrFlow.js and also to the use-case and the code here.

### Features
- Interactive generation of (parameterized) test data
- Interactive training and validation of (parameterized) Neural Network
- Ease of extension of interactive parameters (sliders) thanks to a high level of abstraction and automation
- Encapulation, reusability, and separation of concerns with a purely class-based code
    - Immutable implementation of logic (but not data models) 

### Use-Case
- Samples of a decaying exponential function (with controllable sampling rate, noise, sparsity ...) are defined as ```x``` and ```y``` vectors.
- An Artificial Neural Network is trained to learn the dependency ```y=f(x)``` with controllable learning parameters.
- The plots ```x vs. y``` and ```x vs. f(x)``` are superimposed for a semi-quantitative assessment of the learning.

### Execution
- Browser client
  - Double-click on the file [index.html](./src/html/index.html) (or open it with your browser of choice). No server is needed.
- Node.js client
  - Run the command ```npm install``` and then ```npm test```.
  - An example of this client is the [GitHub Action](./.github/workflows/node.js.yml) of this project.  
- Deno client
  - Run the command ```Deno test ./tests/javascript/deno/test.js --allow-read```.
  - An example of this client is the [GitHub Action](./.github/workflows/deno.yml) of this project.
	- If the main client is a Deno-client, it is recommended that the main code be imported in the test file. The current implementation is based on the restriction to support 3 different clients.

### CI/CD
 - GitHub Action for [node.js](./.github/workflows/node.js.yml)
 - GitHub Action for [Deno](./.github/workflows/deno.yml)

### Snapshot of Control-Parameters and Outputs
  The sliders for ```epochs``` (250 vs. 400) and ```noise``` (2.5% vs. 5%) provide a dynamic and user-defined compromise between <i>accuracy</i> and <i>speed</i> (Note that these stochastic results are reproducible only in a statistical sense).
  
  ![](./assets/Sliders.png)

### Defining new sliders
In the function ```Controller.createSliders```, simply extend the local variable ```sliders```. For example to include a slider for ```sparsity```, the definition can be extended as follows:
```
const sliders = {
  noisePercent: Controller.createSlider('Noise', 0, inputDataParameters.noisePercent * 2, sliderIndex++),
  epochs: Controller.createSlider('Epoch', 100, inputDataParameters.epochs * 2, sliderIndex++),
  sparsity: Controller.createSlider('Sparsity', 1, 6, sliderIndex++)
};
``` 
All other steps, including the setting of callbacks, will be done automatically.
