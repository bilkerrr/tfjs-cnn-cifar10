import {Cifar10} from './data.js';

async function load () {
    const data = new Cifar10()
    await data.load()
  
    const [train_images, train_labels] = data.nextTrainBatch();
    const [test_images, test_labels] = data.nextTestBatch();
    console.log(train_images, train_labels, test_images, test_labels);

    const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'}); 
    for(let i=0; i<10; i++) {
        const imageTensor = tf.tidy(() => {
            return train_images.slice([i, 0], [1, train_images.shape[1]]).reshape([32, 32, 3]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }

    const model = tf.sequential({layers: [
        tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', inputShape: [32, 32, 3]}),
        tf.layers.maxPooling2d({poolSize: [2, 2]}),
        tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu'}),
        tf.layers.maxPooling2d({poolSize: [2, 2]}),
        tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu'}),
        tf.layers.flatten(),
        tf.layers.dense({units: 64, activation: 'relu'}),
        tf.layers.dense({units: 10, activation: 'softmax'})
    ]});
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);

    model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});

    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const history = await model.fit(train_images.reshape([50000, 32, 32, 3]), train_labels, {
        validationData: [test_images.reshape([10000, 32, 32, 3]), test_labels], 
        batchSize: 1,
        epochs: 10,
        callbacks: fitCallbacks
    });

    tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);

    await model.save('downloads://my-model')
}

load();