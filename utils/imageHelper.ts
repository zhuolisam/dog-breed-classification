import axios from 'axios';

import { Tensor } from 'onnxruntime-web';

export async function getImageTensorFromPath(
  path: HTMLImageElement,
  dims: number[] = [1, 3, 224, 224]
): Promise<Tensor> {
  // 1. load the image
  var image = await loadImageFromPath(path, dims[2], dims[3]);
  // 2. convert to tensor
  var imageTensor = imageDataToTensor(image, dims);
  // 3. return the tensor
  return imageTensor;
}

async function loadImageFromPath(
  img: HTMLImageElement,
  width: number = 224,
  height: number = 224
): Promise<Buffer> {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  // resize image
  canvas.width = width;
  canvas.height = canvas.width * (img.height / img.width);

  // draw scaled image
  // @ts-ignore
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  
  // document.getElementById('scaled-image').src = canvas.toDataURL();

  // return data
  // @ts-ignore
  return ctx.getImageData(0, 0, width, width).data;
}

function imageDataToTensor(
  image: any,
  dims: number[],
  transform = {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  }
): Tensor {
  // 1. Get buffer data from image and create R, G, and B arrays.
  var imageBufferData = image;
  const [redArray, greenArray, blueArray] = new Array(
    new Array<number>(),
    new Array<number>(),
    new Array<number>()
  );

  // 2. Loop through the image buffer and extract the R, G, and B channels

  // normalize the image with mean and std according to the rgb channel
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(
      (imageBufferData[i] / 255.0 - transform.mean[0]) / transform.std[0]
    );
    greenArray.push(
      (imageBufferData[i + 1] / 255.0 - transform.mean[1]) / transform.std[1]
    );
    blueArray.push(
      (imageBufferData[i + 2] / 255.0 - transform.mean[2]) / transform.std[2]
    );
    // skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. convert to float32
  let i,
    l = transposedData.length; // length, we need this for the loop
  // create the Float32Array size 3 * 224 * 224 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }
  // 5. create the tensor object from onnxruntime-web.
  const inputTensor = new Tensor('float32', float32Data, dims);
  return inputTensor;
}
