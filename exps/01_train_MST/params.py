# audio parameters
sr = 22050
sequence_time = 1.0
sequence_hop_time = 1.0
audio_hop = 512
audio_win = 1024
n_fft = 1024
normalize_data = 'minmax' # MST paper
get_annotations = True
mel_bands = 128
htk = True
normalize_energy = False

# training
learning_rate = 0.001 # MST paper
epochs = 101
batch_size = 64
sed_early_stopping = 100
epoch_limit = None
fit_verbose = True
fine_tuning = False

#model
large_cnn = True
frames = False # False for MST

label_list = (['air_conditioner', 'car_horn', 'children_playing',
               'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer', 'siren', 'street_music'])         

# Create output folders
expfolder = '../../sed_endtoend/mst'

audio_folder = '/data_ssd/users/pzinemanas/maestria/URBAN-SED/audio22050'
label_folder='/data_ssd/users/pzinemanas/maestria/URBAN-SED/annotations'

dataset = 'URBAN-SED'