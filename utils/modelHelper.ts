import * as ort from 'onnxruntime-web';
import _ from 'lodash';
import { class_names } from './predict';

export async function runModel(
  model_path: string,
  preprocessedData: any
): Promise<[any, number, number]> {
  // Create session and set options. See the docs here for more options:
  //https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
  const session = await ort.InferenceSession.create(
    model_path,
    { executionProviders: ['webgl'], graphOptimizationLevel: 'all' }
  );
  // Run inference and get results.
  var [results, probs, inferenceTime] = await runInference(session, preprocessedData);
  return [results, probs, inferenceTime];
}

async function runInference(
  session: ort.InferenceSession,
  preprocessedData: any
): Promise<[any, number, number]> {
  // Get start time to calculate inference time.
  const start = new Date();
  // create feeds with the input name from model export and the preprocessed data.
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;

  // Run the session inference.
  const outputData = await session.run(feeds);
  // Get the end time to calculate inference time.
  const end = new Date();
  // Convert to seconds.
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;
  // Get output results with the output name from the model export.
  const output = outputData[session.outputNames[0]];
  //Get the softmax of the output data. The softmax transforms values to be between 0 and 1
  var outputSoftmax = softmax(Array.prototype.slice.call(output.data));

  // console.log('wtf-> ', outputSoftmax)
  // Get the class name from the class_names array.
  var [propMax, propMaxIndex] = argMax(outputSoftmax);
  var result = class_names[propMaxIndex];

  return [result, propMax, inferenceTime];
}

//The softmax transforms values to be between 0 and 1
function softmax(resultArray: number[]): any {
  // Get the largest value in the array.
  const largestNumber = Math.max(...resultArray);
  // Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
  const sumOfExp = resultArray
    .map((resultItem) => Math.exp(resultItem - largestNumber))
    .reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
  //Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
  return resultArray.map((resultValue, index) => {
    return Math.exp(resultValue - largestNumber) / sumOfExp;
  });
}

function argMax(arr: number[]) {
  let max = arr[0];
  let maxIndex = 0;
  for (var i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }
  return [max, maxIndex];
}

/**
 * Find top k imagenet classes
 */
// export function imagenetClassesTopK(classProbabilities: any, k = 5) {
//   const probs =
//       _.isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities;

//   const sorted = _.reverse(_.sortBy(probs.map((prob: any, index: number) => [prob, index]), (probIndex: Array<number> ) => probIndex[0]));

//   const topK = _.take(sorted, k).map((probIndex: Array<number>) => {
//     const iClass = imagenetClasses[probIndex[1]];
//     return {
//       id: iClass[0],
//       index: parseInt(probIndex[1].toString(), 10),
//       name: iClass[1].replace(/_/g, ' '),
//       probability: probIndex[0]
//     };
//   });
//   return topK;
// }
