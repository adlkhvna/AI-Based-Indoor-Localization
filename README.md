# AI-Based-Indoor-Localization
Indoor Localization is the process of obtaining the location of a user or a device in an indoor environment. In my project I used the signal strength from multiple WI-FI access points (WAPs) as the input to predict a longitude and latitude coordinates. 

Used UJIIndoorLoc’s training set and a validation sets.
Each dataset contains:
520 WAP features: These are the RSSI values from 520 different Wi-Fi access points.
Longitude and Latitude: The ground-truth coordinates of the device.
Negative integer values ranges between -104dBm (extremely poor signal) to 0dbM. 
The positive value 100 is used to denote when a WAP was not detected. 

No signal" values (recorded as 100) were replaced with a very weak signal value (-105).
All signal strengths were scaled to a range between 0 and 1 using MinMaxScaler.

Feature engineering process follows the methodology proposed by Mittal et al. Used the Pearson Correlation Coefficient to measure the linear relationship between each WAP's signal strength and the device's location. 
Implementation:
Calculated the Pearson correlation for each of the 520 WAP features.
Used the Hadamard Product to multiply the scaled WAP features by their corresponding correlation coefficients. 

Reshaped 1D list of 520 features into a 2D image.

To predict the location for a new set of RSSI values, the k-NN algorithm finds the 5 most similar WAP data points in the training set and averages their known locations.

Two Conv2D layers with MaxPooling and Dropout to learn features and prevent overfitting.
A Flatten layer to convert the 2D feature maps into a 1D vector.
A Dense hidden layer.
An output Dense layer with 2 neurons (for Longitude and Latitude).

Evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Euclidean Distance Error. 

KNN outperformed better than CNN
