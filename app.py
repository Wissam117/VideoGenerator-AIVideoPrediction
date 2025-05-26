import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/videos"
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Define ALL custom classes that were used during training
@tf.keras.utils.register_keras_serializable(package="Custom")
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        # Define layers
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.add = layers.Add()
        
        # Add skip connection if needed
        self.skip_connection = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.skip_connection = layers.Conv2D(self.filters, 1, padding='same')
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.skip_connection is not None:
            inputs = self.skip_connection(inputs)
            
        x = self.add([inputs, x])
        return self.relu2(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class ST_LSTM_Cell(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ST_LSTM_Cell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        # Define convolutional layers
        self.conv_xh = tf.keras.layers.Conv2D(
            filters=4 * filters,
            kernel_size=kernel_size,
            padding='same'
        )
        self.conv_m = tf.keras.layers.Conv2D(
            filters=3 * filters,
            kernel_size=kernel_size,
            padding='same'
        )
        self.conv_o = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same'
        )

    def call(self, x_t, h_prev, c_prev, m_prev):
        # Compute gates
        xh = self.conv_xh(tf.concat([x_t, h_prev], axis=-1))
        i_xh, f_xh, g_xh, o_xh = tf.split(xh, num_or_size_splits=4, axis=-1)

        m = self.conv_m(m_prev)
        i_m, f_m, m_m = tf.split(m, num_or_size_splits=3, axis=-1)

        # Update states
        i_t = tf.sigmoid(i_xh + i_m)
        f_t = tf.sigmoid(f_xh + f_m)
        g_t = tf.tanh(g_xh)
        c_t = f_t * c_prev + i_t * g_t

        m_t = f_t * m_prev + i_t * tf.tanh(m_m)

        o_t = tf.sigmoid(o_xh + self.conv_o(m_t))
        h_t = o_t * tf.tanh(c_t)

        return h_t, c_t, m_t

    def get_config(self):
        config = super(ST_LSTM_Cell, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class PredRNNLayer(tf.keras.layers.Layer):
    def __init__(self, filters, num_layers, input_frames, output_frames, kernel_size=(3, 3), **kwargs):
        super(PredRNNLayer, self).__init__(**kwargs)
        self.filters = filters
        self.num_layers = num_layers
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.kernel_size = kernel_size
        # Create ST-LSTM cells for each layer
        self.stlstm_cells = [ST_LSTM_Cell(filters, kernel_size) for _ in range(num_layers)]
        # Output convolution
        self.conv_output = tf.keras.layers.Conv2D(
            filters=3,  # Assuming RGB output
            kernel_size=(1, 1),
            activation='sigmoid',
            padding='same'
        )

    def call(self, inputs, training=None):
        # inputs.shape: (batch_size, input_frames, height, width, channels)
        input_frames = self.input_frames
        output_frames = self.output_frames
        total_frames = input_frames + output_frames

        # Get static shape dimensions if possible
        batch_size = tf.shape(inputs)[0]
        height = inputs.shape[2]
        width = inputs.shape[3]
        channels = inputs.shape[4]

        # Handle dynamic height and width
        if height is None:
            height = tf.shape(inputs)[2]
        if width is None:
            width = tf.shape(inputs)[3]

        # Initialize states
        h_t = [tf.zeros((batch_size, height, width, self.filters)) for _ in range(self.num_layers)]
        c_t = [tf.zeros((batch_size, height, width, self.filters)) for _ in range(self.num_layers)]
        m_t = tf.zeros((batch_size, height, width, self.filters))

        outputs = []

        for t in range(total_frames):
            if t < input_frames:
                x_t = inputs[:, t]
            else:
                x_t = x_pred  # Use last predicted frame

            h_t_prev = h_t.copy()
            c_t_prev = c_t.copy()
            m_t_prev = m_t

            for l in range(self.num_layers):
                if l == 0:
                    h, c, m_t = self.stlstm_cells[l](x_t, h_t_prev[l], c_t_prev[l], m_t_prev)
                else:
                    h, c, m_t = self.stlstm_cells[l](h_t[l-1], h_t_prev[l], c_t_prev[l], m_t)
                h_t[l] = h
                c_t[l] = c

            x_pred = self.conv_output(h_t[-1])

            if t >= input_frames:
                outputs.append(x_pred)

        outputs = tf.stack(outputs, axis=1)
        return outputs

    def get_config(self):
        config = super(PredRNNLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'num_layers': self.num_layers,
            'input_frames': self.input_frames,
            'output_frames': self.output_frames,
            'kernel_size': self.kernel_size
        })
        return config

# Define the custom loss function that was used during training
def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss

class DataGenerator(Sequence):
    def __init__(self, video_file, input_frames=10, output_frames=5, batch_size=1, img_size=(64, 64)):
        """
        Initialization for video file input (direct video processing).
        
        Parameters:
        - video_file: The path to the input video file.
        - input_frames: Number of frames to use as input.
        - output_frames: Number of frames to predict.
        - batch_size: Number of samples per batch.
        """
        self.video_file = video_file
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.batch_size = batch_size
        self.img_size = img_size
        self.frames = self._extract_frames_from_video(video_file)
        self.indexes = np.arange(len(self.frames) - self.input_frames - self.output_frames)
        self.on_epoch_end()
        print(f"Initialized DataGenerator with {len(self.frames)} frames.")

    def _extract_frames_from_video(self, video_file):
        """
        Extract frames from a video file.
        """
        frames = []
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame to the target size and convert to grayscale (to match training)
            frame_resized = cv2.resize(frame, self.img_size)
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray[:, :, None]  # Add channel dimension
            frames.append(frame_gray)
        cap.release()
        return np.array(frames)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.floor(len(self.frames) / self.batch_size))

    def on_epoch_end(self):
        """
        Shuffles indexes after each epoch.
        """
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X_batch, Y_batch = [], []
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        for idx in batch_indexes:
            X = self.frames[idx:idx + self.input_frames]
            Y = self.frames[idx + self.input_frames:idx + self.input_frames + self.output_frames]
            X_batch.append(X)
            Y_batch.append(Y)
        
        X_batch = np.array(X_batch).astype(np.float32) / 255.0  # Normalize
        Y_batch = np.array(Y_batch).astype(np.float32) / 255.0  # Normalize
        
        return X_batch, Y_batch


def load_models():
    """
    Loads all pre-trained models with proper custom objects.
    Returns:
        convlstm_model, predrnn_model, vit_model
    """
    # Enable unsafe deserialization for Lambda layers
    tf.keras.config.enable_unsafe_deserialization()
    
    # Define all custom objects that might be used in any of the models
    custom_objects = {
        'ResidualBlock': ResidualBlock,
        'PredRNNLayer': PredRNNLayer,
        'ST_LSTM_Cell': ST_LSTM_Cell,
        'custom_loss': custom_loss,
    }
    
    # Define the model paths
    convlstm_path = 'models/convlstm_best_model1.keras'
    predrnn_path = 'models/predrnn_best_model.keras'
    vit_path = 'models/vit.keras'  # or 'models/final_vit_video_model.keras'
    
    try:
        # Load models with custom objects
        convlstm_model = load_model(convlstm_path, custom_objects=custom_objects)
        print("ConvLSTM model loaded successfully")
    except Exception as e:
        print(f"Error loading ConvLSTM model: {e}")
        convlstm_model = None
    
    try:
        predrnn_model = load_model(predrnn_path, custom_objects=custom_objects)
        print("PredRNN model loaded successfully")
    except Exception as e:
        print(f"Error loading PredRNN model: {e}")
        predrnn_model = None
    
    try:
        vit_model = load_model(vit_path, custom_objects=custom_objects)
        print("ViT model loaded successfully")
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        vit_model = None
    
    return convlstm_model, predrnn_model, vit_model


def generate_video(model, input_frames, output_frames, test_gen, video_filename, img_size=(64, 64)):
    """
    Generate video showing input frames followed by predicted frames.
    """
    try:
        X_test, Y_true = test_gen.__getitem__(0)
        
        if X_test.size == 0 or Y_true.size == 0:
            return None
        
        Y_pred = model.predict(X_test)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 10
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (img_size[1], img_size[0]))
        
        # Write input frames (convert from grayscale to BGR)
        for t in range(input_frames):
            frame = (X_test[0, t] * 255).astype(np.uint8)
            if frame.shape[-1] == 1:  # Grayscale
                frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        
        # Write predicted frames (convert from grayscale to BGR)
        for t in range(output_frames):
            frame = (Y_pred[0, t] * 255).astype(np.uint8)
            if frame.shape[-1] == 1:  # Grayscale
                frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        
        video_writer.release()
        return video_filename
    except Exception as e:
        print(f"Error generating video: {e}")
        return None


# Load models at startup
convlstm_model, predrnn_model, vit_model = load_models()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    model_choice = request.form['model']
    video_file = request.files['video']
    
    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
        # Create a data generator for the input video
        test_gen = DataGenerator(video_path, input_frames=10, output_frames=5, batch_size=1)
        
        # Generate the video using the selected model
        video_output_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'generated_{filename}')
        
        if model_choice == 'convlstm':
            generated_video = generate_video(convlstm_model, 10, 5, test_gen, video_output_filename)
        elif model_choice == 'predrnn':
            generated_video = generate_video(predrnn_model, 10, 5, test_gen, video_output_filename)
        elif model_choice == 'visiontransformer':
            generated_video = generate_video(vit_model, 10, 5, test_gen, video_output_filename)
        else:
            return "Invalid model selection"
        
        if generated_video:
            # Send the video as a downloadable file
            return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(generated_video), as_attachment=True)
        else:
            return "Error in video generation"
    return "Invalid file format"


if __name__ == '__main__':
    app.run(debug=True)