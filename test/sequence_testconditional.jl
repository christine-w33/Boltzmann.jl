using Boltzmann

#for predicting sequences using crbms

input = [1 0 0 0 0;
         1 1 0 0 0;
         0 1 1 0 0;
         0 0 1 1 0;
         0 0 0 1 1;
         0 0 0 0 1]

input = float(input)

sequence = [1; 1; 0; 0; 0; 0;
            0; 1; 1; 0; 0; 0;
            0; 0; 1; 1; 0; 0;
            0; 0; 0; 1; 1; 0;
            0; 0; 0; 0; 1; 1]

test = [1; 1; 0; 0; 0; 0]

sequence = reshape(sequence, 30, 1)
test = reshape(test, 6, 1)

model = fit_sequence(sequence, 6, 5, 50, 5)
prediction = predict_sequence(model, test, 5)
println(prediction)

#Code for further testing

function hamming_distance(test, input)
    error = 0.0
    n = size(input, 1)
    m = size(input, 2)
    for i in 1:n
        for j in 1:m
            if test[i, j] != input[i, j]
                error += 1.0
            end
        end
    end
    error
end

#hamming_averages = []
#stds = []
#for s in 0:0.1:1
#    hamming_total = 0.0
#    hamming_distances = Array(Float64, 1)
#    for i in 1:5
#        model = fit_sequence(sequence, 6, 5, 100, 5, s)
#        #forecast = predict(model, test; n_gibbs=20)
        #println(forecast)
#        prediction = predict_sequence(model, test, 5)
        #hamming_distance = hamming_distance(prediction, input)
#        hamming_distances = push!(hamming_distances, hamming_distance(prediction, input))
#        hamming_total += hamming_distance(prediction, input)
#    end
#    hamming_average = hamming_total/5
#    hamming_averages = push!(hamming_averages, hamming_average)
#    stds = push!(stds, std(hamming_distances))
#end
#println("errors: ", hamming_averages)
#println("standard deviations: ", stds)
