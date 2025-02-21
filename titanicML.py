import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------- Phase 1

# Load the training dataset
training_data = pd.read_csv('train.csv')

# Preprocess the training data
## Drop columns that won't be used in prediction
training_data = training_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

## Handle categorical variables
### Models can't directly understand text directly, they work with numbers
### I used get_dummies to convert categorical columns into numerical columns using one-hot-encoding
### I included drop_first to avoid redundancy
training_data = pd.get_dummies(training_data, columns=['Sex', 'Embarked'], drop_first=True)

## Handle missing values
training_data['Age'].fillna(training_data['Age'].median(), inplace=True)
training_data['Fare'].fillna(training_data['Age'].median(), inplace=True)

# Define predictors and target
## Include all relevant features except survived. This ensures the model learns from the input features, otherwise it would be just memorizing the outcome
predictors = training_data.drop(columns=['Survived']).values
## Transformed this column to a categorical type, for keras classification using softmax output
target = to_categorical(training_data['Survived'])

# Standardize the predictors
## Make the features uniform in scale (no feature is disproportionatley large or small)
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)

# Get the number of columns in predictors
n_cols = predictors.shape[1]

# Build the model
## Using a sequential model
model = Sequential()
## Add the input layer, specifying the number of nodes, the activation function, 
## and the number of columns of features 'must', leaving it open for rows
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
## Add the first hidden layer
model.add(Dense(50, activation='relu'))
## Add the second hidden layer
model.add(Dense(50, activation='relu'))
## Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
## Using the sgd optimizer, and categorical_corssentropy as the loss function
## While monitoring the accuracy metric to evaluate how the model is performing
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Specify the early stopping
# patience defines how many epochs the model can go without improving before we stop training
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
## Training the model , using 50 epochs , saving 20% of the training data for validation,
## and using 32 samples before updating the weights
model.fit(predictors,target, epochs=50, validation_split=0.2, batch_size=32, callbacks=[early_stopping_monitor])



# Load the test data 
test_data = pd.read_csv('test.csv')

passenger_ids = test_data['PassengerId']

# Preprocess the test data 
test_data = test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Age'].median(), inplace=True)
test_data = scaler.transform(test_data)

# Make predictions using the model
predictions = model.predict(test_data)

# Get the predicted probabilities of survival
predicted_prob_true = predictions[:, 1]

# Interpreting the results

## Tranform the result using a threshold
predicted_classes = (predicted_prob_true > 0.5).astype(int)
print(predicted_classes)


submission_df = pd.DataFrame({
    'PassengerId' : passenger_ids,
    'Survived' : predicted_classes
})

submission_df.to_csv('submission.csv', index=False)

# ------------------------------------------------------- Phase 2

# Changing optimization parameters

# Create list of learning rates: lr_to_test
lr_to_test = [.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer=my_optimizer,loss='categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors,target)









