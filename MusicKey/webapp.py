from flask import Flask, render_template, request
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid conflicts with Flask
import matplotlib.pyplot as plt

app = Flask(__name__)

class Tonal_Fragment:
    def __init__(self, waveform, sr):
        self.waveform = waveform
        self.sr = sr
        self.chromagram = librosa.feature.chroma_cqt(y=self.waveform, sr=self.sr)

    def detect_key(self):
        chroma_mean = np.mean(self.chromagram, axis=1)
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_index = np.argmax(chroma_mean)
        key = pitch_classes[key_index]
        mode = 'Major' if chroma_mean[(key_index + 3) % 12] > chroma_mean[(key_index + 4) % 12] else 'Minor'
        return key, mode

    def print_chromagram(self):
        fig, ax = plt.subplots(figsize=(12, 4))
        librosa.display.specshow(self.chromagram, sr=self.sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1, ax=ax)
        ax.set_title('Chromagram')
        fig.colorbar(ax.get_children()[0], ax=ax)
        plt.tight_layout()
        plt.savefig('static/chromagram.png')  # Save the figure to a file
        plt.close(fig)  # Close the figure to free up memory

    def get_frequent_notes(self):
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        top_indices = np.argsort(np.mean(self.chromagram, axis=1))[-7:]
        frequent_notes = [pitch_classes[i] for i in top_indices]
        return frequent_notes

def handle_upload():
    uploaded_file = request.files['file']
    waveform, sr = librosa.load(uploaded_file)
    tonal_fragment = Tonal_Fragment(waveform, sr)
    detected_key, mode = tonal_fragment.detect_key()
    tonal_fragment.print_chromagram()
    frequent_notes = tonal_fragment.get_frequent_notes()
    return detected_key, mode, frequent_notes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    detected_key, mode, frequent_notes = handle_upload()
    return render_template('upload_result.html', detected_key=detected_key, mode=mode, frequent_notes=frequent_notes)

if __name__ == '__main__':
    app.run(debug=True)
