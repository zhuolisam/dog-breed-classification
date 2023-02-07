## 70 Dog Breed Classification
This is a [Next.js](https://nextjs.org) app that classifies 70 different dog breeds, trained with Efficient Net B0, achieved ~90% accuracy.

The model is trained using PyTorch, can be viewed in `/pytorch-training` folder.

After training, the model is saved as ONNX format. The model is loaded and inferenced on edge using ONNX Runtime Web API, thus has a **low edge inference time**.

## Tech Used
WebApp
* Next.js
* Tailwind
* ONNX Runtime Web API
* Edge Inference

Machine Learning Model
* PyTorch
* ResNet18 and EffNet B0
* Kaggle Dataset - [here](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)


## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
