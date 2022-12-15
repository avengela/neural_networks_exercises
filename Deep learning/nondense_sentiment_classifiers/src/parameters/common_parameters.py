# osadzenie przestrzeni wektorowej
n_dimensions = 64
pad_type = trunc_type = 'pre'

# dropout
drop_embed = 0.2

# fitting
loss = "binary_crossentropy"
optimizer = "nadam"
metrics = ["accuracy"]
batch_size = 128
