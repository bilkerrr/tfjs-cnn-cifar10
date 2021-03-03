// CIFAR10 dataset for TensorFlow.js
// based on https://github.com/zqingr/tfjs-cifar10
// and https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/

const IMG_WIDTH = 32;
const IMG_HEIGHT = 32;
const NUM_CLASSES = 10
const DATA_PRE_NUM = 10000
const IMAGE_SIZE =  IMG_WIDTH * IMG_HEIGHT * 3 
const TRAIN_IMAGES = [
    'cifar10/data_batch_1.png',
    'cifar10/data_batch_2.png',
    'cifar10/data_batch_3.png',
    'cifar10/data_batch_4.png',
    'cifar10/data_batch_5.png'
]
const TEST_IMAGES = [
    'cifar10/test_batch.png'
]
const TRAIN_LABLES = 'cifar10/train_lables.json'
const TEST_LABLES = 'cifar10/test_lables.json'

export class Cifar10 {
  
    constructor() {
    }
    
    loadImg (src) {
        return new Promise((resolve, reject) => {
          const img = new Image();
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          img.src = src
          img.onload = () => {
            canvas.width = img.naturalWidth
            canvas.height = img.naturalHeight
    
            const datasetBytesBuffer = new ArrayBuffer(canvas.width * canvas.height * 3 * 4)
            const datasetBytesView = new Float32Array(datasetBytesBuffer)
    
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height)
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
            for (let j = 0, i = 0; j < imageData.data.length; j++) {
              if ((j + 1) % 4 === 0) continue
              datasetBytesView[i++] = imageData.data[j] / 255
            }
    
            resolve(datasetBytesView)
          }
          img.onerror = reject
        })
    }
    
    loadImages (srcs) {
        return Promise.all(srcs.map(this.loadImg))
    }
    
    async load () {
        this.trainDatas = await this.loadImages(TRAIN_IMAGES)
        this.testDatas = await this.loadImages(TEST_IMAGES)
    
        this.trainLables = await (await fetch(TRAIN_LABLES)).json()
        this.testLables = await (await fetch(TEST_LABLES)).json()
    
        this.trainM = this.trainLables.length
        this.testM = this.testLables.length
    
        this.trainIndices = tf.util.createShuffledIndices(this.trainM)
        this.testIndices = tf.util.createShuffledIndices(this.testM)

        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    nextBatch (batchSize, data, labels, index) {
      const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE)  
      const batchLables = []
  
      for (let i = 0; i < batchSize; i++) {
        const idx = index()
        const currentIdx = idx % DATA_PRE_NUM
        const dataIdx = Math.floor(idx / DATA_PRE_NUM)
  
        const image = data[dataIdx].slice(currentIdx * IMAGE_SIZE, currentIdx * IMAGE_SIZE + IMAGE_SIZE)
        batchImagesArray.set(image, i * IMAGE_SIZE)
        batchLables.push(labels[idx])
      }
      const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE])
      const ys = tf.oneHot(batchLables, NUM_CLASSES)
  
      return [ xs, ys ]
    }
  
    nextTrainBatch (batchSize) {
        if(!batchSize)
            batchSize = this.trainM;

        this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
    
        return this.nextBatch(batchSize, this.trainDatas, this.trainLables, () => {
            this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
            return this.trainIndices[this.shuffledTrainIndex]
        })
    }
  
    nextTestBatch (batchSize) {
        if(!batchSize)
            batchSize = this.testM;

        this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length
    
        return this.nextBatch(
            batchSize, this.testDatas, this.testLables, () => {
            this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length
            return this.testIndices[this.shuffledTestIndex]
            })
    }
  }