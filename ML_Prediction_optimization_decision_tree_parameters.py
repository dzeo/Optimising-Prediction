# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt


#..c:\\users\\acer\\desktop\\pythonjoe\\git_hub\\ML_Prediction_Optimisation_decision_tree\\
# Path of the file to read
file_path = 'train.csv'

home_data = pd.read_csv(file_path)

# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
model = DecisionTreeRegressor(random_state=1)
# Fit Model
model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))



#This function gets the MAE of any given leaf number automaically
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
import operator
mae_dic = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    mae_dic[max_leaf_nodes] = mae
    
    print("Max_leaf_nodes: %d  \t\t MAE : %d " % (max_leaf_nodes, mae))
#print(mae_dic)

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(mae_dic.items(), key= operator.itemgetter(1))[0]
#print(best_tree_size)


#here, we plot candidate_max_leaf_nodes against its corresponding MAE value
plt.plot(mae_dic.keys(), mae_dic.values())
plt.show()

#STEP 2: Fit Model Using All data

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 0)

# fit the final model and uncomment the next two lines
final_model.fit(X,y)
