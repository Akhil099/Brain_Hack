using Flux
using JLD2
using Microstructure
using MLUtils

abc =load("/Users/akhil_rao/Downloads/toy_training_set.jld2")
abc["labels"]
abc["inputs"]
using Flux: crossentropy, onecold, onehotbatch, train!, params
using LinearAlgebra, Random, Statistics
inputs = abc["inputs"]
labels = abc["labels"]

mean_input = mean(inputs, dims = 2)
std_inputs = std(inputs, dims = 2)
normalized_input = (inputs .-mean_input) ./std_inputs

#Splitting the datasets to train and test datasets
train_indices, val_indices = splitobs(collect(1:size(inputs, 2)), at=0.8)  # 80-20 split

train_inputs = normalized_input[:, train_indices]
train_labels = labels[:, train_indices]

val_inputs = normalized_input[:, val_indices]
val_labels = labels[:, val_indices]

Model = Chain(
    Dense(size(train_inputs, 1), 64, relu),
    Dense(64, 8,relu), 
    Dense(8, 6, relu)
)
loss(x, y) = Flux.crossentropy(Model(x), y, dims = 1)
optimizer = RMSProp()
println("Model output size: ", size(Model(train_inputs)))
println("Label size: ", size(train_labels))

function log_loss()
    return() ->println("Current loss: ", loss(train_inputs, train_labels))
end


Flux.train!(loss, params(Model), [(train_inputs, train_labels)], optimizer, cb = Flux.throttle(log_loss(), 10))
