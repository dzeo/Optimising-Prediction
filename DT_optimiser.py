import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import operator


file_path = 'train.csv'

home_data = pd.read_csv(file_path)

# Create target
y = home_data.SalePrice
# Create Features
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
print(f"Validation MAE: {val_mae.round()}")




def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Gets the associated  MeanAbsoluteError for a  given leaf number for 
    the Deccision tree regressor.
    """
    
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [i for i in range(2,500,25)]
mae_dic = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    mae_dic[max_leaf_nodes] = mae
    
    print(f"Max_leaf_nodes: {max_leaf_nodes}  \t\t MAE : {mae} " )

# Store the best value of max_leaf_nodes (which is either of 5, 25, 50, 100, 250 or 500)
best_tree_size = min(mae_dic.items(), key= operator.itemgetter(1))[0]
print("best tree size",best_tree_size)

#here, we plot candidate_max_leaf_nodes against its corresponding MAE value
plt.plot(list(mae_dic.keys()), list(mae_dic.values()))
plt.xlabel("Number of leaves")
plt.ylabel("Mean abosolute error")
plt.title("Plot of Number of Leaves against the Mean Absolute Error")
plt.show()

#STEP 2: Fit Model Using All data

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 0)

# fit the final model 
final_model.fit(X,y)

