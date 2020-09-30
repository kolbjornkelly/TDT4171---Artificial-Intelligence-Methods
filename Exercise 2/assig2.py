import numpy

transitions = numpy.array([[0.7, 0.3], [0.3, 0.7]])
sensor_true = numpy.array([[0.9, 0],[0, 0.2]])
sensor_false = numpy.array([[0.1, 0],[0, 0.8]])

observations = [True, True, False, True, True]


def normalize(vec):
    arr = numpy.array(vec)
    return arr/arr.sum()

def predict(vec):
    return transitions.dot(vec)

def update(prediction_vec, day):
    umbrella = observations[day-1]
    if umbrella:
        return normalize(sensor_true.dot(prediction_vec))
    else:
        return normalize(sensor_false.dot(prediction_vec))

def forward(vec, day):
    vec = predict(vec)
    vec = update(vec, day)
    return vec

def backward(vec, day):
    umbrella = observations[day-1]
    if umbrella:
        return transitions.dot((sensor_true).dot(vec))
    else:
        return transitions.dot((sensor_false).dot(vec))


forward_vec = [0]*6
backward_vec = [0]*6
forward_vec[0] = [0.5, 0.5] # Initial belief
b = numpy.array([1, 1])

for i in range(1,6): # Go forward
    forward_vec[i] = forward(forward_vec[i-1], i)
for i in range(5,-1,-1): # Go back again
    backward_vec[i] = normalize(numpy.multiply(forward_vec[i], b))
    b = backward(b, i)
    print(b)
