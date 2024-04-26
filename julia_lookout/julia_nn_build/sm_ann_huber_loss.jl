using Flux
using JLD2
using Microstructure
using MLUtils
using BSON

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
loss(x, y) = Flux.huber_loss(Model(x), y; delta = 0.05) # changes behaviour as |ŷ - y| > δ
optimizer = RMSProp()
println("Model output size: ", size(Model(train_inputs)))
println("Label size: ", size(train_labels))

function log_loss()
    return() ->println("Current loss: ", loss(train_inputs, train_labels))
end


Flux.train!(loss, params(Model), [(train_inputs, train_labels)], optimizer, cb = Flux.throttle(log_loss(), 10))

BSON.@save "model_huberloss.bson" Model
BSON.@load "model_huberloss.bson" Model

function huber_loss(predictions, targets; delta = 0.05)
    return (Flux.Losses.huber_loss(predictions, targets; delta = delta))
end

# function mae_loss(predictions, target)
#     return mean(Flux.Losses.mae_loss(predictions, target))
# end

predictions = Model(val_inputs)

val_rmse = losses_rmse(predictions, val_labels)
println("Rmse loss on test_data: ", val_rmse)

test_huber_loss = huber_loss(predictions, val_labels)
println("Huber loss on test_data: ", test_huber_loss)

println("Mae loss in the test_data: ", Flux.Losses.mae(predictions, val_labels))
println("mse loss in the test_data: ", (Flux.Losses.mse(predictions, val_labels)))

###1. compare how different loss functions affect training performance

# 1. train on the training dataset using loss function 1 => mlp
#  apply this model to validation data to get predict labels
# evaluation metric should be one but the loss functions can be 3 or more






#between line 10 and 13 to check the data
# print(size(abc["labels"]))
# print(size(abc["inputs"]))
# X_train_raw, y_train_raw = abc.labels()

#between line 15 and 17 to check the data
# println("First few inputs: ", inputs[:, 1:5])
# println("First few labels: ", labels[:, 1:5])]