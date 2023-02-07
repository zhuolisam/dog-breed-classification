import { getImageTensorFromPath } from './imageHelper';
import { runModel } from './modelHelper';

export const class_names = [
  'Afghan',
  'African Wild Dog',
  'Airedale',
  'American Hairless',
  'American Spaniel',
  'Basenji',
  'Basset',
  'Beagle',
  'Bearded Collie',
  'Bermaise',
  'Bichon Frise',
  'Blenheim',
  'Bloodhound',
  'Bluetick',
  'Border Collie',
  'Borzoi',
  'Boston Terrier',
  'Boxer',
  'Bull Mastiff',
  'Bull Terrier',
  'Bulldog',
  'Cairn',
  'Chihuahua',
  'Chinese Crested',
  'Chow',
  'Clumber',
  'Cockapoo',
  'Cocker',
  'Collie',
  'Corgi',
  'Coyote',
  'Dalmation',
  'Dhole',
  'Dingo',
  'Doberman',
  'Elk Hound',
  'French Bulldog',
  'German Sheperd',
  'Golden Retriever',
  'Great Dane',
  'Great Perenees',
  'Greyhound',
  'Groenendael',
  'Irish Spaniel',
  'Irish Wolfhound',
  'Japanese Spaniel',
  'Komondor',
  'Labradoodle',
  'Labrador',
  'Lhasa',
  'Malinois',
  'Maltese',
  'Mex Hairless',
  'Newfoundland',
  'Pekinese',
  'Pit Bull',
  'Pomeranian',
  'Poodle',
  'Pug',
  'Rhodesian',
  'Rottweiler',
  'Saint Bernard',
  'Schnauzer',
  'Scotch Terrier',
  'Shar_Pei',
  'Shiba Inu',
  'Shih-Tzu',
  'Siberian Husky',
  'Vizsla',
  'Yorkie',
];

export async function inference(
  path: HTMLImageElement
): Promise<[any, number, number]> {
  // 1. Convert image to tensor
  const imageTensor = await getImageTensorFromPath(path);

  // Specify path to use
  const resnet_path = '/model/resnet_dog_breed.onnx';
  const effnet_path = '/model/effnet_dog_breed.onnx';

  // 2. Run model
  const [predictions, probs, inferenceTime] = await runModel(
    effnet_path,
    imageTensor
  );
  // 3. Return predictions and the amount of time it took to inference.
  return [predictions, probs, inferenceTime];
}
