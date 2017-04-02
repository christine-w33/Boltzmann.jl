using Boltzmann



X = [1 0 0 0 0;
     1 1 0 0 0;
     0 1 1 0 0;
     0 0 1 1 0;
     0 0 0 1 1;
     0 0 0 0 1]

X = float(X)
data = [1 1 0 0 0 0]
data = reshape(data, 6, 1)

layers = [("ECCA", BernoulliRBM(6, 10)),
          ("CADG", ConditionalRBM(Bernoulli, Bernoulli, 10, 20; steps=1))]

#          ("CADG", BernoulliRBM(Bernoulli, Bernoulli, 10, 20; steps = 1))]

dbn = DBN(layers)

fit(dbn, X)
println(dbn_predict(dbn, data, 5))
